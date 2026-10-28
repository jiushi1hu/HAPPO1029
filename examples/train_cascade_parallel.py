"""Five-year/process-parallel orchestration used by train_cascade.py."""
import json
from pathlib import Path
import subprocess
from contextlib import nullcontext
import hashlib
import numpy as np
from harl.envs.cascade_reservoir.episodes import episode_environment, positive_integer, training_windows
from harl.envs.cascade_reservoir.terminal_storage import TerminalReleaseBounds
from harl.envs.cascade_reservoir.training_metrics import log_episode
from harl.runners.cascade_parallel_runner import CascadeParallelRunner

ROOT = Path(__file__).resolve().parents[1]


def check_episode(data, physical, years, days, bins):
    env = episode_environment(data, physical, years, days, bins)
    bounds = {} if env.seasonal is None else dict(storage_lower=env.storage_lower, storage_upper=env.storage_upper)
    TerminalReleaseBounds(env.data, env.physics.dt, env.storage, env.data.external[env.start:env.stop],
                          env.terminal_storage_target, **bounds).check_feasible()


def run_parallel_experiment(data, physical, config, args):
    days = positive_integer(config['train']['episode_length'], 'episode_days')
    count = positive_integer(config['train']['n_rollout_threads'], 'parallel_envs')
    episodes = positive_integer(config['experiment']['episodes'], 'episodes')
    if config['experiment'].get('fixed_training_year') is not None:
        raise ValueError('Use annual collection for fixed-training-year diagnostics')
    if config['experiment']['rollout_episodes'] != count or episodes % count:
        raise ValueError('Parallel rollout_episodes must equal parallel_envs, and episodes must be divisible by it')
    windows = training_windows(data, days)
    bins = config['model']['action_bins']
    validation = list(data.validation_years)
    selection = 'validation' if validation else 'final_training_update'
    print('Checking {} training windows{} ...'.format(len(windows),
        ' and the held-out validation sequence' if validation else ''), flush=True)
    for years in list(windows.values())+([validation] if validation else []):
        check_episode(data, physical, years, days, bins)
    metadata = dict(record_type='parallel_configuration', collection_mode='five_year_parallel',
        episode_horizon_days=days, parallel_environments=count, collector_backend='spawn worker processes',
        episodes_per_seed=episodes, updates_per_seed=episodes//count,
        collector_steps_per_update=days*count, collector_steps_per_seed=days*episodes,
        training_window_starts=list(windows), training_window_source_years=windows,
        validation_source_years=validation, test_source_years=list(data.test_years),
        validation_enabled=bool(validation), policy_selection=selection,
        calendar='fixed consecutive daily steps; leap days retained; no annual resets; '
                 '1825 steps can end before the fifth calendar anniversary',
        terminal_storage_mode=physical.get('terminal_storage_mode', 'free'),
        terminal_target_scope='end of the entire episode only; seasonal hard deadlines still apply each year',
        evaluation=('one optional continuous validation sequence selects the policy; ' if validation else
                    'no validation evaluations or checkpoint ranking; use the final trained policy; ')+
                   'final representative test years are concatenated in configured order '
                   'with storage carried between them',
        model_scope='paper collection layout only; reward, observation and optimizer definitions remain explicit below',
        config=config, physical_config=physical, data_hashes=data.hashes,
        source_code_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(list((ROOT/'harl').rglob('*.py'))+list((ROOT/'examples').glob('*.py'))+
                            list((ROOT/'harl/configs').rglob('*.yaml')))})
    print(json.dumps(metadata, indent=2), flush=True)
    if args.check_data_only:
        print('Multi-year data and physical feasibility checks passed; no workers or training started.', flush=True)
        return
    if not args.no_plot:
        subprocess.run([args.plot_python, str(ROOT/'examples/plot_cascade.py'), '--check-backend'], check=True)
    stream = None
    if args.episode_log is not None:
        args.episode_log.parent.mkdir(parents=True, exist_ok=True)
        stream = args.episode_log.open('x', encoding='utf-8')
    with (stream if stream is not None else nullcontext()):
        log_episode(metadata, stream)
        runners = []
        for seed in config['experiment']['seeds']:
            runner = CascadeParallelRunner(data, physical, config, seed, episode_log=stream)
            runner.run(episodes, config['experiment']['eval_interval'])
            runners.append(runner)
        tests = []
        if not args.skip_final_test:
            if not data.test_years:
                raise ValueError('Final testing requires a nonempty held-out test sequence')
            check_episode(data, physical, list(data.test_years), days, bins)
            tests = [row for runner in runners for row in runner.final_test()]
        selected = [runner.test_policy_metadata() for runner in runners]
        result = dict(record_type='parallel_run_complete', seeds=config['experiment']['seeds'],
            episode_horizon_days=days, parallel_environments=count,
            episodes_per_seed=episodes, training_steps_per_seed=[r.training_steps for r in runners],
            completed_episodes_per_seed=[r.completed_episodes for r in runners],
            updates_per_seed=[r.update_count for r in runners],
            validation_enabled=bool(validation), policy_selection=selection,
            selected_episodes=[row['selected_episode'] for row in selected],
            selected_training_steps=[row['selected_training_steps'] for row in selected],
            selected_updates=[row['selected_update'] for row in selected],
            final_test_skipped=bool(args.skip_final_test), tests=tests)
        if validation:
            result.update(best_episodes=[r.best_iteration for r in runners],
                          best_training_steps=[r.best_training_steps for r in runners])
        log_episode(result, stream)
        if not args.no_plot:
            payload = dict(episode_horizon_days=days, gamma=config['algo']['gamma'],
                seeds=config['experiment']['seeds'], curve_source='validation' if validation else 'training',
                histories=[r.validation_history if validation else r.training_history for r in runners])
            subprocess.run([args.plot_python, str(ROOT/'examples/plot_cascade_parallel.py')],
                           input=json.dumps(payload), text=True, check=True)
    print('Finished parallel collection experiment. Models remain in memory; '
          'JSONL metrics are saved only when --episode-log is supplied.', flush=True)
