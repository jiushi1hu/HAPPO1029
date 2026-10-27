"""Paired in-sample check of policy optimization budgets; no test-year tuning."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from examples.train_cascade import load_configuration
from harl.algorithms.actors.happo import validate_policy_optimization, validate_kl_rollback
from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.evaluation import evaluate_policy, validate_operating_feasibility
from harl.runners.cascade_reservoir_runner import (
    CascadeReservoirRunner, validate_fixed_training_year, validate_seed)


def resolve_candidate_epochs(requested, kl_only):
    epochs = (5 if kl_only else 20) if requested is None else requested
    epochs, _ = validate_policy_optimization(epochs, None)
    if kl_only and epochs != 5:
        raise ValueError('--kl-only requires five epochs in both arms')
    return epochs


def _state_fingerprint(value):
    """Hash numeric state contents, independently of storage addresses."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            visit(item.detach().cpu().contiguous().numpy())
        elif isinstance(item, np.ndarray):
            digest.update(('array:'+item.dtype.str+':'+repr(item.shape)+':').encode())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            digest.update(b'dict{')
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                visit(key); visit(item[key])
            digest.update(b'}')
        elif isinstance(item, (tuple, list)):
            digest.update((type(item).__name__+'[').encode())
            for entry in item:
                visit(entry)
            digest.update(b']')
        else:
            digest.update((type(item).__name__+':'+repr(item)+';').encode())
    visit(value)
    return digest.hexdigest()


def actor_state_fingerprint(runner):
    return _state_fingerprint([actor.actor.state_dict() for actor in runner.actor])


def training_state_fingerprint(runner):
    """Learning state only: exclude treatment settings, histories and wall time."""
    return _state_fingerprint(dict(
        actors=[a.actor.state_dict() for a in runner.actor],
        actor_optimizers=[a.actor_optimizer.state_dict() for a in runner.actor],
        critic=runner.critic.critic.state_dict(),
        critic_optimizer=runner.critic.critic_optimizer.state_dict(),
        value_normalizer=None if runner.value_normalizer is None else runner.value_normalizer.state_dict(),
        actor_modes=[a.actor.training for a in runner.actor], critic_mode=runner.critic.critic.training,
        torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate(),
        year_rng=runner.year_rng.get_state(), iteration=runner.iteration,
        update_count=runner.update_count, training_steps=runner.training_steps))


def _statistics(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Comparison requires finite nonempty one-dimensional values')
    sd = float(values.std(ddof=1)) if len(values) > 1 else None
    return dict(mean=float(values.mean()), sd=sd,
                se=None if sd is None else sd/float(np.sqrt(len(values))))


def paired_kl_result(baseline, guarded):
    """Compare final current policies on the same per-year action-seed grid."""
    before, after = baseline['final_policy_report'], guarded['final_policy_report']
    keys = ('seed', 'gamma', 'scope', 'evaluation_years', 'sampled_repeats', 'evaluation_seeds',
            'episode', 'update', 'training_steps')
    if (any(k not in before or k not in after for k in keys) or
            baseline['seed'] != guarded['seed'] or before.get('seed') != baseline['seed'] or
            any(before.get(k) != after.get(k) for k in keys)):
        raise ValueError('Paired arms must use the same seed, gamma, scope and evaluation grid')
    count = before['sampled_repeats']
    values = [np.asarray(r['stochastic_repeat_means_reward'], dtype=np.float64) for r in (before, after)]
    if (isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1 or
            any(v.shape != (count,) or not np.isfinite(v).all() for v in values)):
        raise ValueError('Paired repeat values must match the recorded positive repeat count')
    differences = values[1]-values[0]
    stats = _statistics(differences)
    saved = guarded['actor_epochs_saved']
    return dict(record_type='paired_kl_comparison', seed=baseline['seed'],
        comparison='guarded minus baseline at the final current policy; no best-checkpoint selection',
        uncertainty_scope='paired action sampling conditional on this training seed and evaluation year',
        initial_state_match=baseline['initial_training_state_hash']==guarded['initial_training_state_hash'],
        final_state_match=baseline['final_training_state_hash']==guarded['final_training_state_hash'],
        final_actor_state_match=baseline['final_actor_state_hash']==guarded['final_actor_state_hash'],
        deterministic_reward_difference=after['deterministic_mean_reward']-before['deterministic_mean_reward'],
        sampled_repeat_reward_differences=differences.tolist(),
        sampled_reward_difference_mean=stats['mean'], sampled_reward_difference_sd=stats['sd'],
        sampled_reward_difference_se=stats['se'], guarded_epochs_saved_by_actor=saved,
        guarded_threshold_crossings_by_actor=guarded['kl_early_stops'],
        guarded_rejected_steps_by_actor=guarded.get('actor_optimizer_rejected_steps', [0.]*len(saved)),
        no_optimizer_steps_saved=not any(saved),
        baseline_actor_optimizer_steps=baseline['actor_optimizer_steps'],
        guarded_actor_optimizer_steps=guarded['actor_optimizer_steps'])


def kl_comparison_summary(pairs):
    if not pairs or len({p['seed'] for p in pairs}) != len(pairs):
        raise ValueError('One paired result per unique training seed is required')
    return dict(record_type='paired_kl_summary', seeds=[p['seed'] for p in pairs],
        sampled_reward_difference_across_seeds=_statistics([p['sampled_reward_difference_mean'] for p in pairs]),
        deterministic_reward_difference_across_seeds=_statistics([p['deterministic_reward_difference'] for p in pairs]),
        uncertainty_scope='variation of paired estimated differences across training seeds; not per-day samples',
        all_initial_states_match=all(p['initial_state_match'] for p in pairs),
        all_final_states_match=all(p['final_state_match'] for p in pairs),
        all_final_actor_states_match=all(p['final_actor_state_match'] for p in pairs),
        total_guarded_epochs_saved=int(sum(sum(p['guarded_epochs_saved_by_actor']) for p in pairs)),
        total_guarded_rejected_steps=int(sum(sum(p.get('guarded_rejected_steps_by_actor', [])) for p in pairs)),
        total_guarded_threshold_crossings=int(sum(sum(p['guarded_threshold_crossings_by_actor']) for p in pairs)))


class _ArmLog:
    """Tag the runner's flushed JSONL records without touching existing logs."""
    def __init__(self, output, arm):
        self.output, self.arm = output, arm

    def write(self, text):
        for line in text.splitlines():
            self.output.write(json.dumps(dict(json.loads(line), arm=self.arm), allow_nan=False)+'\n')

    def flush(self):
        self.output.flush()


def _sampled_summary(report):
    penalties = [row['episode_penalty'] for year in report['sampled_results_by_year'] for row in year['repeats']]
    return dict(mean_reward=report['stochastic_mean_reward'], mean_TWh=report['stochastic_mean_TWh'],
                mean_penalty=float(np.mean(penalties)),
                repeat_rewards=report['stochastic_repeat_means_reward'],
                repeat_sd=report['stochastic_repeat_sd_reward'], repeat_se=report['stochastic_repeat_se_reward'])


def sampled_check(runner, repeats):
    """Common evaluation seeds, isolated from training and optimizer RNGs."""
    rows = []
    for repeat in range(repeats):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1000000 + runner.seed*1000 + repeat)
            rows.append(evaluate_policy(runner.actor, runner.data, runner.physical_config,
                runner.fixed_training_year, runner.seed, runner.iteration, runner.training_steps,
                deterministic=False, gamma=runner.algo_args['algo']['gamma']))
    return dict(mean_reward=float(np.mean([r['episode_reward'] for r in rows])),
                mean_TWh=float(np.mean([r['energy_TWh'] for r in rows])),
                mean_penalty=float(np.mean([r['episode_penalty'] for r in rows])),
                repeat_rewards=[r['episode_reward'] for r in rows],
                mean_compliance={k:float(np.mean([r['service_shortfalls'][k]['compliance_rate'] for r in rows]))
                                 for k in ('guaranteed_output', 'ecological_flow')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--year', type=int, default=2013)
    parser.add_argument('--updates', type=int, default=6)
    parser.add_argument('--rollout-episodes', type=int, default=6)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1])
    parser.add_argument('--evaluation-repeats', type=int, default=3)
    parser.add_argument('--kl-only', action='store_true',
                        help='Matched five-epoch experiment: change only target_kl and retain step-1 diagnostics')
    parser.add_argument('--candidate-ppo-epochs', type=int,
                        help='Default: 20 for the legacy budget comparison, 5 with --kl-only')
    parser.add_argument('--candidate-target-kl', type=float, default=.01)
    parser.add_argument('--candidate-kl-rollback', action='store_true',
                        help='Enable per-step rejection/restoration in the guarded arm; use --kl-only to keep both epoch budgets at five')
    parser.add_argument('--output', type=Path, required=True, help='New JSONL comparison file')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be a new file')
    if min(args.updates, args.rollout_episodes, args.evaluation_repeats) < 1:
        parser.error('updates, rollout episodes and evaluation repeats must be positive')
    try:
        args.candidate_ppo_epochs = resolve_candidate_epochs(args.candidate_ppo_epochs, args.kl_only)
        args.candidate_ppo_epochs, args.candidate_target_kl = validate_policy_optimization(
            args.candidate_ppo_epochs, args.candidate_target_kl)
        validate_kl_rollback(args.candidate_kl_rollback, args.candidate_target_kl)
        args.seeds = [validate_seed(seed) for seed in args.seeds]
        if len(set(args.seeds)) != len(args.seeds):
            raise ValueError('seeds must be unique')
    except ValueError as exc:
        parser.error(str(exc))
    config, physical = load_configuration()
    paired_diagnostics = args.kl_only or args.candidate_kl_rollback
    config['algo']['kl_rollback'] = False  # The experiment baseline always disables KL protection.
    config['algo']['joint_target_kl'] = None  # This driver compares individual actor guards only.
    config['model']['action_bins'] = 50
    config['experiment'].update(fixed_training_year=args.year, rollout_episodes=args.rollout_episodes,
                               episodes=args.updates*args.rollout_episodes,
                               seeds=args.seeds, eval_interval=1,
                               diagnostic_eval_episodes=[0,args.updates*args.rollout_episodes],
                               diagnostic_stochastic_repeats=args.evaluation_repeats,
                               diagnostics=paired_diagnostics)
    data = CascadeData(ROOT/'data/cascade_reservoir_new', physical, config['experiment'])
    validate_fixed_training_year(args.year, data)
    validate_operating_feasibility(data, physical, years=[args.year])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as output:
        metadata = dict(record_type='paired_check_configuration', year=args.year,
            comparison_type=('kl_rollback' if args.candidate_kl_rollback else
                             'kl_only' if args.kl_only else 'optimization_budget'),
            changed_settings=(['target_kl'] if args.kl_only else ['ppo_epoch', 'target_kl'])+
                             (['kl_rollback'] if args.candidate_kl_rollback else []),
            kl_rollback_by_arm=dict(baseline=False, guarded=args.candidate_kl_rollback),
            epoch_accounting='completed epochs require all minibatches; optimizer counts describe partially accepted epochs',
            kl_stopping_semantics=('check each optimizer step; restore actor/Adam/gradients and stop on crossing' if
                args.candidate_kl_rollback else 'check after each actor epoch; keep the threshold-crossing update'),
            evaluation_scope='in-sample training-year diagnostic; no generalization claim',
            updates=args.updates, annual_episodes=args.updates*args.rollout_episodes,
            rollout_episodes=args.rollout_episodes, seeds=args.seeds,
            evaluation_repeats=args.evaluation_repeats, gamma=config['algo']['gamma'],
            gae_lambda=config['algo']['gae_lambda'], action_bins=config['model']['action_bins'],
            actor_lr=config['model']['lr'], critic_lr=config['model']['critic_lr'],
            critic_epochs=config['algo']['critic_epoch'],
            arms=dict(baseline=dict(ppo_epochs=5, target_kl=None),
                      guarded=dict(ppo_epochs=args.candidate_ppo_epochs, target_kl=args.candidate_target_kl)),
            reward_penalties=physical['reward_penalties'], source_hashes=data.hashes,
            common_config=config, physical_config=physical,
            source_code_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(list((ROOT/'harl').rglob('*.py'))+
                                list((ROOT/'harl/configs').rglob('*.yaml'))+
                                [Path(__file__).resolve(), ROOT/'examples/train_cascade.py'])})
        output.write(json.dumps(metadata)+'\n'); output.flush()
        pairs = []
        for seed in args.seeds:
            results = []
            for name, epochs, kl in [('baseline', 5, None),
                                    ('guarded', args.candidate_ppo_epochs, args.candidate_target_kl)]:
                current = copy.deepcopy(config)
                current['algo'].update(ppo_epoch=epochs, target_kl=kl,
                    kl_rollback=name=='guarded' and args.candidate_kl_rollback)
                runner = CascadeReservoirRunner(data, physical, current, seed,
                    **({'episode_log':_ArmLog(output,name)} if paired_diagnostics else {}))
                initial_hash = training_state_fingerprint(runner)
                if results and results[0]['initial_training_state_hash'] != initial_hash:
                    raise RuntimeError('Paired arms did not start from identical learning state')
                print('PAIRED_ARM_START '+json.dumps(dict(arm=name,seed=seed,
                    initial_training_state_hash=initial_hash)),flush=True)
                if not paired_diagnostics:
                    initial = sampled_check(runner, args.evaluation_repeats)
                    print('PAIRED_CHECK_INITIAL '+json.dumps(dict(arm=name, seed=seed, **initial)), flush=True)
                started = time.monotonic()
                runner.run(args.updates*args.rollout_episodes, 1)
                if paired_diagnostics:
                    initial_report, final_report = runner.fixed_year_diagnostics[0], runner.fixed_year_diagnostics[-1]
                    initial, final = _sampled_summary(initial_report), _sampled_summary(final_report)
                else:
                    final = sampled_check(runner, args.evaluation_repeats)
                row = dict(record_type='paired_check_result', arm=name, seed=seed,
                    initial_sampled=initial, final_sampled=final,
                    deterministic_history=runner.fixed_year_history,
                    actor_optimizer_steps=[sum(r['actor{}_optimizer_steps'.format(i)]
                                               for r in runner.training_history) for i in range(5)],
                    actor_optimizer_attempts=[sum(r['actor{}_optimizer_attempts'.format(i)]
                                               for r in runner.training_history) for i in range(5)],
                    actor_optimizer_rejected_steps=[sum(r['actor{}_optimizer_rejected_steps'.format(i)]
                                               for r in runner.training_history) for i in range(5)],
                    kl_early_stops=[sum(r['actor{}_kl_early_stopped'.format(i)]
                                       for r in runner.training_history) for i in range(5)],
                    training_steps=runner.training_steps, elapsed_seconds=time.monotonic()-started,
                    initial_training_state_hash=initial_hash,
                    final_training_state_hash=training_state_fingerprint(runner),
                    final_actor_state_hash=actor_state_fingerprint(runner))
                if paired_diagnostics:
                    row.update(initial_policy_report=initial_report, final_policy_report=final_report,
                        sampled_years=runner.sampled_years,
                        actor_epochs_saved=[sum(max(0.,epochs-r['actor{}_ppo_epochs_completed'.format(i)])
                            if r['actor{}_kl_early_stopped'.format(i)] else 0.
                            for r in runner.training_history) for i in range(5)],
                        training_history=runner.training_history,
                        policy_evaluation_history=runner.fixed_year_diagnostics,
                        training_diagnostics=runner.diagnostics_history)
                results.append(row)
                output.write(json.dumps(row,allow_nan=False)+'\n'); output.flush()
                print('PAIRED_CHECK '+json.dumps({k:v for k,v in row.items()
                      if k in ('arm','seed','initial_sampled','final_sampled','actor_optimizer_steps',
                               'actor_optimizer_attempts','actor_optimizer_rejected_steps',
                               'kl_early_stops','actor_epochs_saved','elapsed_seconds')}), flush=True)
            if paired_diagnostics:
                pair = paired_kl_result(*results)
                pairs.append(pair)
                output.write(json.dumps(pair,allow_nan=False)+'\n'); output.flush()
                print('PAIRED_KL '+json.dumps(pair),flush=True)
        if pairs:
            summary = kl_comparison_summary(pairs)
            output.write(json.dumps(summary,allow_nan=False)+'\n'); output.flush()
            print('PAIRED_KL_SUMMARY '+json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
