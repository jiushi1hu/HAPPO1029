"""Training-only paired experiments for GAE lambda or annual rollout batch size."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from examples.train_cascade import load_configuration
from examples.compare_cascade_policy_budget import (
    _ArmLog, _statistics, actor_state_fingerprint, training_state_fingerprint)
from harl.algorithms.actors.happo import (
    validate_policy_optimization, validate_kl_rollback, validate_joint_target_kl)
from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.diagnostics import policy_evaluation_report
from harl.envs.cascade_reservoir.evaluation import evaluate_policy, validate_operating_feasibility
from harl.runners.cascade_reservoir_runner import (
    CascadeReservoirRunner, validate_gae_lambda, validate_rollout_episodes, validate_seed)


def comparison_settings(config, intervention, candidate_gae_lambda=None,
                        candidate_rollout_episodes=None):
    """Change exactly one setting, leaving optimization budgets otherwise fixed."""
    baseline = dict(gae_lambda=validate_gae_lambda(config['algo']['gae_lambda']),
                    rollout_episodes=validate_rollout_episodes(config['experiment']['rollout_episodes']))
    candidate = dict(baseline)
    if intervention == 'gae_lambda':
        if not config['algo'].get('use_gae', False):
            raise ValueError('the gae_lambda intervention requires use_gae=True')
        if candidate_rollout_episodes is not None:
            raise ValueError('candidate rollout episodes require the rollout_episodes intervention')
        candidate['gae_lambda'] = validate_gae_lambda(
            1. if candidate_gae_lambda is None else candidate_gae_lambda)
    elif intervention == 'rollout_episodes':
        if candidate_gae_lambda is not None:
            raise ValueError('candidate GAE lambda requires the gae_lambda intervention')
        candidate['rollout_episodes'] = validate_rollout_episodes(
            12 if candidate_rollout_episodes is None else candidate_rollout_episodes)
        if candidate['rollout_episodes'] <= baseline['rollout_episodes']:
            raise ValueError('candidate rollout episodes must exceed the baseline batch size')
    else:
        raise ValueError('intervention must be gae_lambda or rollout_episodes')
    if candidate == baseline:
        raise ValueError('candidate setting must differ from baseline')
    return baseline, candidate


def year_schedule(years, episodes, seed):
    """A common year stream independent of action/optimizer RNG and batching."""
    if not years or len(set(years)) != len(years):
        raise ValueError('Training years must be nonempty and unique')
    episodes = validate_rollout_episodes(episodes)
    seed = validate_seed(seed)
    return [int(year) for year in np.random.RandomState(seed).choice(years, size=episodes)]


def evaluate_training_endpoint(runner, years, previous_report=None):
    """Observe current policies without selecting checkpoints or visiting holdouts."""
    if (not years or len(set(years)) != len(years) or
            not set(years).issubset(runner.data.training_years) or
            set(years).intersection(runner.data.validation_years) or
            set(years).intersection(runner.data.test_years)):
        raise ValueError('Evaluation years must be unique training years, excluding all holdouts')
    modes = [(actor.actor, actor.actor.training) for actor in runner.actor]
    initial_state = training_state_fingerprint(runner)
    try:
        with torch.random.fork_rng(devices=[]):
            rows = [evaluate_policy(runner.actor, runner.data, runner.physical_config,
                year, runner.seed, runner.iteration, runner.training_steps,
                gamma=runner.algo_args['algo']['gamma']) for year in years]
            report = policy_evaluation_report(runner, rows, previous_report=previous_report,
                                              scope='training_years')
    finally:
        for module, training in modes:
            module.train(training)
    if training_state_fingerprint(runner) != initial_state:
        raise RuntimeError('Endpoint evaluation changed the training state')
    report['evaluation_preserved_training_state'] = True
    return report


class TrainingSignalRunner(CascadeReservoirRunner):
    """Use training-only endpoints instead of validation/checkpoint selection."""

    def __init__(self, *args, evaluation_years, annual_episodes, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluation_years = list(evaluation_years)
        self.comparison_episodes = annual_episodes
        self.policy_reports = []

    def evaluate_checkpoint(self):
        if self.iteration not in (0, self.comparison_episodes):
            return None
        report = evaluate_training_endpoint(self, self.evaluation_years,
            previous_report=self.policy_reports[-1] if self.policy_reports else None)
        self.policy_reports.append(report)
        if self.episode_log is not None:
            self.episode_log.write(json.dumps(report, allow_nan=False)+'\n')
            self.episode_log.flush()
        return report['deterministic_mean_reward']


def paired_learning_result(baseline, candidate):
    """Pair action repeats while allowing the batch intervention's update counts to differ."""
    before, after = baseline['final_policy_report'], candidate['final_policy_report']
    grid = ('seed', 'gamma', 'scope', 'evaluation_years', 'sampled_repeats',
            'evaluation_seeds', 'episode', 'training_steps')
    if (baseline['seed'] != candidate['seed'] or baseline['seed'] != before.get('seed') or
            any(key not in before or key not in after or before[key] != after[key] for key in grid) or
            before.get('scope') != 'training_years'):
        raise ValueError('Paired arms require matching training seed, annual budget and evaluation grid')
    if (baseline['sampled_years'] != candidate['sampled_years'] or
            len(baseline['sampled_years']) != before['episode'] or
            baseline['training_steps'] != candidate['training_steps'] or
            baseline['training_steps'] != before['training_steps']):
        raise ValueError('Paired arms require the same annual year schedule and collected steps')
    if baseline['initial_training_state_hash'] != candidate['initial_training_state_hash']:
        raise ValueError('Paired arms must start from identical learning state')
    intervention = baseline['intervention']
    settings = baseline['settings'], candidate['settings']
    if (intervention not in ('gae_lambda', 'rollout_episodes') or
            candidate['intervention'] != intervention or
            any(set(setting) != {'gae_lambda','rollout_episodes'} for setting in settings) or
            {key for key in settings[0] if settings[0][key] != settings[1][key]} != {intervention}):
        raise ValueError('Paired arms must change exactly the named learning-signal setting')
    for setting in settings:
        validate_gae_lambda(setting['gae_lambda'])
        validate_rollout_episodes(setting['rollout_episodes'])
    if intervention == 'gae_lambda' and before['update'] != after['update']:
        raise ValueError('The GAE lambda intervention must retain the same update budget')
    count = before['sampled_repeats']
    values = [np.asarray(report['stochastic_repeat_means_reward'], dtype=np.float64)
              for report in (before, after)]
    if (isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1 or
            any(value.shape != (count,) or not np.isfinite(value).all() for value in values)):
        raise ValueError('Paired reward repeats must match the positive repeat count')
    differences = values[1]-values[0]
    stats = _statistics(differences)
    return dict(record_type='paired_learning_signal_comparison', seed=baseline['seed'],
        intervention=intervention,
        comparison='candidate minus baseline at the final current policy; no checkpoint selection',
        uncertainty_scope='paired action sampling conditional on these training years and this trained seed',
        initial_state_match=baseline['initial_training_state_hash']==candidate['initial_training_state_hash'],
        final_state_match=baseline['final_training_state_hash']==candidate['final_training_state_hash'],
        final_actor_state_match=baseline['final_actor_state_hash']==candidate['final_actor_state_hash'],
        annual_episodes=before['episode'], training_steps=before['training_steps'],
        baseline_updates=before['update'], candidate_updates=after['update'],
        equal_update_budgets=before['update']==after['update'], year_schedule_match=True,
        baseline_settings=settings[0], candidate_settings=settings[1],
        deterministic_reward_difference=after['deterministic_mean_reward']-before['deterministic_mean_reward'],
        sampled_repeat_reward_differences=differences.tolist(),
        sampled_reward_difference_mean=stats['mean'], sampled_reward_difference_sd=stats['sd'],
        sampled_reward_difference_se=stats['se'],
        baseline_actor_optimizer_steps=baseline['actor_optimizer_steps'],
        candidate_actor_optimizer_steps=candidate['actor_optimizer_steps'])


def learning_comparison_summary(pairs):
    if (not pairs or len({pair['seed'] for pair in pairs}) != len(pairs) or
            len({pair['intervention'] for pair in pairs}) != 1):
        raise ValueError('A summary requires one paired result per unique seed for one intervention')
    return dict(record_type='paired_learning_signal_summary',
        intervention=pairs[0]['intervention'], seeds=[pair['seed'] for pair in pairs],
        sampled_reward_difference_across_seeds=_statistics([p['sampled_reward_difference_mean'] for p in pairs]),
        deterministic_reward_difference_across_seeds=_statistics([p['deterministic_reward_difference'] for p in pairs]),
        uncertainty_scope='variation of paired estimated differences across trained seeds; not per-day samples',
        all_initial_states_match=all(pair['initial_state_match'] for pair in pairs),
        all_year_schedules_match=all(pair['year_schedule_match'] for pair in pairs),
        all_update_budgets_equal=all(pair['equal_update_budgets'] for pair in pairs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intervention', required=True, choices=('gae_lambda', 'rollout_episodes'))
    parser.add_argument('--episodes', type=int, default=12, help='Same annual episode budget in both arms')
    parser.add_argument('--rollout-episodes', type=int, help='Baseline annual batch size; also used by the lambda candidate')
    parser.add_argument('--candidate-gae-lambda', type=float)
    parser.add_argument('--candidate-rollout-episodes', type=int)
    parser.add_argument('--training-years', nargs='+', type=int, default=[2012,2013],
                        help='Training and endpoint evaluation years; excludes validation/test years')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1])
    parser.add_argument('--evaluation-repeats', type=int, default=3)
    parser.add_argument('--action-bins', type=int, default=50)
    kl = parser.add_mutually_exclusive_group()
    kl.add_argument('--target-kl', type=float)
    kl.add_argument('--no-kl-limit', action='store_true')
    rollback = parser.add_mutually_exclusive_group()
    rollback.add_argument('--kl-rollback', dest='kl_rollback', action='store_true')
    rollback.add_argument('--no-kl-rollback', dest='kl_rollback', action='store_false')
    parser.set_defaults(kl_rollback=None)
    joint = parser.add_mutually_exclusive_group()
    joint.add_argument('--joint-target-kl', type=float)
    joint.add_argument('--no-joint-kl-limit', action='store_true')
    parser.add_argument('--output', type=Path, required=True, help='New JSONL file, never overwritten')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be a new file')
    config, physical = load_configuration()
    try:
        args.episodes = validate_rollout_episodes(args.episodes)
        args.evaluation_repeats = validate_rollout_episodes(args.evaluation_repeats)
        if args.action_bins < 2:
            raise ValueError('action bins must be at least two')
        args.seeds = [validate_seed(seed) for seed in args.seeds]
        if len(set(args.seeds)) != len(args.seeds):
            raise ValueError('seeds must be unique')
        if len(set(args.training_years)) != len(args.training_years):
            raise ValueError('training years must be unique')
        if args.rollout_episodes is not None:
            config['experiment']['rollout_episodes'] = validate_rollout_episodes(args.rollout_episodes)
        baseline, candidate = comparison_settings(config, args.intervention,
            args.candidate_gae_lambda, args.candidate_rollout_episodes)
        if args.intervention == 'rollout_episodes' and args.episodes <= baseline['rollout_episodes']:
            raise ValueError('batch comparison requires enough episodes for more than one baseline update')
        if args.target_kl is not None or args.no_kl_limit:
            config['algo']['target_kl'] = args.target_kl
        if args.kl_rollback is not None:
            config['algo']['kl_rollback'] = args.kl_rollback
        if args.joint_target_kl is not None or args.no_joint_kl_limit:
            config['algo']['joint_target_kl'] = args.joint_target_kl
        config['algo']['ppo_epoch'], config['algo']['target_kl'] = validate_policy_optimization(
            config['algo']['ppo_epoch'], config['algo'].get('target_kl'))
        config['algo']['kl_rollback'] = validate_kl_rollback(
            config['algo'].get('kl_rollback', False), config['algo']['target_kl'])
        config['algo']['joint_target_kl'] = validate_joint_target_kl(
            config['algo'].get('joint_target_kl'), config['algo']['kl_rollback'])
    except ValueError as exc:
        parser.error(str(exc))
    config['model']['action_bins'] = args.action_bins
    config['experiment'].update(training_years=args.training_years, fixed_training_year=None,
        episodes=args.episodes, diagnostics=True, diagnostic_eval_episodes=[],
        diagnostic_stochastic_repeats=args.evaluation_repeats, seeds=args.seeds)
    data = CascadeData(ROOT/'data/cascade_reservoir_new', physical, config['experiment'])
    if (not set(args.training_years).issubset(data.training_years) or
            set(args.training_years).intersection(data.validation_years) or
            set(args.training_years).intersection(data.test_years)):
        parser.error('training years cannot include validation or test years')
    validate_operating_feasibility(data, physical, years=args.training_years)
    schedules = {seed:year_schedule(args.training_years, args.episodes, seed) for seed in args.seeds}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as output:
        source_paths = sorted(list((ROOT/'harl').rglob('*.py'))+
            list((ROOT/'harl/configs').rglob('*.yaml'))+
            [Path(__file__).resolve(), ROOT/'examples/train_cascade.py',
             ROOT/'examples/compare_cascade_policy_budget.py'])
        metadata = dict(record_type='learning_signal_configuration', intervention=args.intervention,
            changed_settings=[args.intervention], annual_episodes=args.episodes,
            arms=dict(baseline=baseline, candidate=candidate), training_years=args.training_years,
            year_schedules=schedules, year_schedule_rule='RandomState(training_seed).choice(training_years, annual_episodes)',
            evaluation_scope='in-sample training years; no validation/test evaluation or checkpoint selection',
            budget_interpretation=('equal annual episodes and updates; only GAE lambda differs' if
                args.intervention=='gae_lambda' else 'equal annual episodes; larger batches reduce update frequency and maximum optimizer-step opportunities; retained steps are logged'),
            seeds=args.seeds, evaluation_repeats=args.evaluation_repeats,
            common_config=config, physical_config=physical, data_hashes=data.hashes,
            source_code_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
        output.write(json.dumps(metadata, allow_nan=False)+'\n'); output.flush()
        pairs = []
        for seed in args.seeds:
            results = []
            for arm, settings in (('baseline', baseline), ('candidate', candidate)):
                current = copy.deepcopy(config)
                current['algo']['gae_lambda'] = settings['gae_lambda']
                current['experiment']['rollout_episodes'] = settings['rollout_episodes']
                runner = TrainingSignalRunner(data, physical, current, seed,
                    episode_log=_ArmLog(output, arm), evaluation_years=args.training_years,
                    annual_episodes=args.episodes)
                schedule_iterator = iter(schedules[seed])
                runner.sample_training_year = lambda: next(schedule_iterator)
                initial_hash = training_state_fingerprint(runner)
                if results and initial_hash != results[0]['initial_training_state_hash']:
                    raise RuntimeError('Paired arms did not start from identical learning state')
                started = time.monotonic()
                runner.run(args.episodes, 1)
                reports = runner.policy_reports
                if runner.sampled_years != schedules[seed] or len(reports) != 2:
                    raise RuntimeError('Comparison failed its year-schedule or endpoint-evaluation contract')
                row = dict(record_type='learning_signal_result', arm=arm, seed=seed,
                    intervention=args.intervention, settings=settings, sampled_years=runner.sampled_years,
                    annual_episodes=runner.iteration, updates=runner.update_count,
                    training_steps=runner.training_steps, elapsed_seconds=time.monotonic()-started,
                    initial_training_state_hash=initial_hash,
                    final_training_state_hash=training_state_fingerprint(runner),
                    final_actor_state_hash=actor_state_fingerprint(runner),
                    initial_policy_report=reports[0], final_policy_report=reports[-1],
                    actor_optimizer_steps=[sum(r['actor{}_optimizer_steps'.format(i)]
                        for r in runner.training_history) for i in range(5)],
                    training_history=runner.training_history, training_diagnostics=runner.diagnostics_history)
                output.write(json.dumps(row, allow_nan=False)+'\n'); output.flush()
                results.append(row)
                print('LEARNING_SIGNAL_ARM '+json.dumps(dict(arm=arm,seed=seed,
                    updates=runner.update_count,training_steps=runner.training_steps,
                    final_sampled_reward=reports[-1]['stochastic_mean_reward'],
                    final_deterministic_reward=reports[-1]['deterministic_mean_reward'])),flush=True)
            pair = paired_learning_result(*results)
            pairs.append(pair)
            output.write(json.dumps(pair, allow_nan=False)+'\n'); output.flush()
            print('PAIRED_LEARNING_SIGNAL '+json.dumps(pair),flush=True)
        summary = learning_comparison_summary(pairs)
        output.write(json.dumps(summary, allow_nan=False)+'\n'); output.flush()
        print('PAIRED_LEARNING_SIGNAL_SUMMARY '+json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
