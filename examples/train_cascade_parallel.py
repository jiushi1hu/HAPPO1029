"""Orchestrate complete multi-year episodes and final held-out evaluation."""
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
import subprocess

from harl.envs.cascade_reservoir.episodes import episode_environment, positive_integer, training_windows
from harl.envs.cascade_reservoir.terminal_storage import TerminalReleaseBounds
from harl.envs.cascade_reservoir.training_metrics import log_episode
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner

ROOT=Path(__file__).resolve().parents[1]


def check_episode(data,physical,years,days,bins):
    env=episode_environment(data,physical,years,days,bins)
    bounds={} if env.seasonal is None else dict(storage_lower=env.storage_lower,storage_upper=env.storage_upper)
    TerminalReleaseBounds(env.data,env.physics.dt,env.storage,env.data.external[env.start:env.stop],
                          env.terminal_storage_target,**bounds).check_feasible()


def run_parallel_experiment(data,physical,config,args):
    days=positive_integer(config['train']['episode_length'],'episode_days')
    count=positive_integer(config['train']['n_rollout_threads'],'parallel_envs')
    episodes=positive_integer(config['experiment']['episodes'],'episodes')
    interval=positive_integer(config['experiment']['log_interval'],'log_interval')
    if episodes % count:
        raise ValueError('episodes must be divisible by parallel_envs')
    windows=training_windows(data,days)
    bins=config['model']['action_bins']
    evaluation_years=list(data.evaluation_years)
    print('Checking {} training windows{} ...'.format(len(windows),
          ' and the evaluation sequence' if not args.skip_final_evaluation else ''),flush=True)
    for years in list(windows.values())+([] if args.skip_final_evaluation else [evaluation_years]):
        check_episode(data,physical,years,days,bins)
    sources=sorted(list((ROOT/'harl').rglob('*.py'))+list((ROOT/'examples').glob('*.py'))+
                   list((ROOT/'harl/configs').rglob('*.yaml')))
    metadata=dict(record_type='run_configuration',collection_mode='five_year_parallel',
        episode_horizon_days=days,parallel_environments=count,collector_backend='spawn worker processes',
        episodes_per_seed=episodes,updates_per_seed=episodes//count,
        collector_steps_per_update=days*count,collector_steps_per_seed=days*episodes,
        training_years=list(data.training_years),training_year_count=len(data.training_years),
        evaluation_source_years=evaluation_years,evaluation_year_count=len(evaluation_years),
        evaluation_labels=list(data.evaluation_labels),training_window_starts=list(windows),
        training_window_source_years=windows,policy_source='final_training_update',
        calendar='fixed consecutive daily steps with leap days retained; storage carries across source-year boundaries',
        terminal_storage_mode=physical['terminal_storage_mode'],
        terminal_target_scope='end of the complete episode; seasonal hard deadlines apply in every source year',
        evaluation='held-out representative years in configured order; storage carries across boundaries; '
                   'one sequence scored after training, without policy ranking',
        config=config,physical_config=physical,data_hashes=data.hashes,
        source_code_hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})
    print(json.dumps(metadata,indent=2,allow_nan=False),flush=True)
    if args.check_data_only:
        print('Multi-year data and physical feasibility checks passed; no workers or training started.',flush=True)
        return
    if not args.no_plot:
        subprocess.run([args.plot_python,str(ROOT/'examples/plot_cascade.py'),'--check-backend'],check=True)
    stream=None
    if args.episode_log is not None:
        args.episode_log.parent.mkdir(parents=True,exist_ok=True)
        stream=args.episode_log.open('x',encoding='utf-8')
    with (stream if stream is not None else nullcontext()):
        log_episode(metadata,stream)
        runners=[]
        for seed in config['experiment']['seeds']:
            runner=CascadeReservoirRunner(data,physical,config,seed,episode_log=stream)
            runner.run(episodes,interval)
            runners.append(runner)
        evaluations=[] if args.skip_final_evaluation else [row for runner in runners for row in runner.final_evaluation()]
        result=dict(record_type='run_complete',seeds=config['experiment']['seeds'],
            episode_horizon_days=days,parallel_environments=count,episodes_per_seed=episodes,
            training_steps_per_seed=[runner.training_steps for runner in runners],
            completed_episodes_per_seed=[runner.completed_episodes for runner in runners],
            updates_per_seed=[runner.update_count for runner in runners],policy_source='final_training_update',
            final_policy_metadata=[runner.evaluation_policy_metadata() for runner in runners],
            evaluation_skipped=bool(args.skip_final_evaluation),evaluations=evaluations)
        log_episode(result,stream)
        if not args.no_plot:
            payload=dict(episode_horizon_days=days,gamma=config['algo']['gamma'],
                seeds=config['experiment']['seeds'],histories=[runner.training_history for runner in runners])
            subprocess.run([args.plot_python,str(ROOT/'examples/plot_cascade_parallel.py')],
                input=json.dumps(payload,allow_nan=False),text=True,check=True)
    print('Finished multi-year experiment. Model persistence/resume is not implemented; '
          'JSONL metrics are saved only with --episode-log.',flush=True)
