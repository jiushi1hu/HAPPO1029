"""Paired in-sample check of policy optimization budgets; no test-year tuning."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from examples.train_cascade import load_configuration
from harl.algorithms.actors.happo import validate_policy_optimization
from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.evaluation import evaluate_policy, validate_operating_feasibility
from harl.runners.cascade_reservoir_runner import (
    CascadeReservoirRunner, validate_fixed_training_year, validate_seed)


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
    parser.add_argument('--candidate-ppo-epochs', type=int, default=20)
    parser.add_argument('--candidate-target-kl', type=float, default=.01)
    parser.add_argument('--output', type=Path, required=True, help='New JSONL comparison file')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be a new file')
    if min(args.updates, args.rollout_episodes, args.evaluation_repeats) < 1:
        parser.error('updates, rollout episodes and evaluation repeats must be positive')
    try:
        args.candidate_ppo_epochs, args.candidate_target_kl = validate_policy_optimization(
            args.candidate_ppo_epochs, args.candidate_target_kl)
        args.seeds = [validate_seed(seed) for seed in args.seeds]
        if len(set(args.seeds)) != len(args.seeds):
            raise ValueError('seeds must be unique')
    except ValueError as exc:
        parser.error(str(exc))
    config, physical = load_configuration()
    config['model']['action_bins'] = 50
    config['experiment'].update(fixed_training_year=args.year, rollout_episodes=args.rollout_episodes,
                               diagnostics=False)
    data = CascadeData(ROOT/'data/cascade_reservoir_new', physical, config['experiment'])
    validate_fixed_training_year(args.year, data)
    validate_operating_feasibility(data, physical, years=[args.year])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as output:
        metadata = dict(record_type='paired_check_configuration', year=args.year,
            evaluation_scope='in-sample training-year diagnostic; no generalization claim',
            updates=args.updates, annual_episodes=args.updates*args.rollout_episodes,
            rollout_episodes=args.rollout_episodes, seeds=args.seeds,
            evaluation_repeats=args.evaluation_repeats, gamma=config['algo']['gamma'],
            gae_lambda=config['algo']['gae_lambda'], action_bins=config['model']['action_bins'],
            actor_lr=config['model']['lr'], critic_lr=config['model']['critic_lr'],
            critic_epochs=config['algo']['critic_epoch'],
            arms=dict(baseline=dict(ppo_epochs=5, target_kl=None),
                      guarded=dict(ppo_epochs=args.candidate_ppo_epochs, target_kl=args.candidate_target_kl)),
            reward_penalties=physical['reward_penalties'], source_hashes=data.hashes)
        output.write(json.dumps(metadata)+'\n'); output.flush()
        for seed in args.seeds:
            for name, epochs, kl in [('baseline', 5, None),
                                    ('guarded', args.candidate_ppo_epochs, args.candidate_target_kl)]:
                current = copy.deepcopy(config)
                current['algo'].update(ppo_epoch=epochs, target_kl=kl)
                runner = CascadeReservoirRunner(data, physical, current, seed)
                initial = sampled_check(runner, args.evaluation_repeats)
                print('PAIRED_CHECK_INITIAL '+json.dumps(dict(arm=name, seed=seed, **initial)), flush=True)
                runner.run(args.updates*args.rollout_episodes, 1)
                final = sampled_check(runner, args.evaluation_repeats)
                row = dict(record_type='paired_check_result', arm=name, seed=seed,
                    initial_sampled=initial, final_sampled=final,
                    deterministic_history=runner.fixed_year_history,
                    actor_optimizer_steps=[sum(r['actor{}_optimizer_steps'.format(i)]
                                               for r in runner.training_history) for i in range(5)],
                    kl_early_stops=[sum(r['actor{}_kl_early_stopped'.format(i)]
                                       for r in runner.training_history) for i in range(5)],
                    training_steps=runner.training_steps)
                output.write(json.dumps(row)+'\n'); output.flush()
                print('PAIRED_CHECK '+json.dumps({k:v for k,v in row.items()
                      if k!='deterministic_history'}), flush=True)


if __name__ == '__main__':
    main()
