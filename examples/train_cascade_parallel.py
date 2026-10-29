"""Orchestrate complete multi-year episodes and final held-out evaluation."""
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
import subprocess

from harl.envs.cascade_reservoir.episodes import episode_environment, positive_integer, training_windows
from harl.envs.cascade_reservoir.training_metrics import log_episode, StepMetricsWriter, STEP_METRIC_FIELDS
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner

ROOT=Path(__file__).resolve().parents[1]


def step_metrics_paths(episode_log, seeds):
    """Keep each seed's compact daily table beside its shared summary log."""
    if episode_log is None:
        return {}
    path=Path(episode_log)
    return {seed:path.with_name(path.stem+'_seed'+str(seed)+'_steps.csv') for seed in seeds}


def check_episode_inputs(data,physical,years,days,bins):
    """Validate the episode view and reset observations, without a future-flow solver."""
    env=episode_environment(data,physical,years,days,bins)
    if len(env.data.dates) != days or env.stop-env.start != days:
        raise ValueError('Episode inputs must contain exactly episode_days daily rows')
    if env.actor_observations().shape != (5,env.observation_size) or env.state().shape != (env.state_size,):
        raise ValueError('Episode observation dimensions do not match the configured layout')


def run_parallel_experiment(data,physical,config,args):
    days=positive_integer(config['train']['episode_length'],'episode_days')
    count=positive_integer(config['train']['n_rollout_threads'],'parallel_envs')
    episodes=positive_integer(config['experiment']['episodes'],'episodes')
    interval=positive_integer(config['experiment']['log_interval'],'log_interval')
    if episodes % count:
        raise ValueError('episodes must be divisible by parallel_envs')
    windows=training_windows(data,days)
    bins=config['model']['action_bins']
    daily_paths=step_metrics_paths(args.episode_log,config['experiment']['seeds'])
    for path in ([] if args.episode_log is None else [args.episode_log])+list(daily_paths.values()):
        if path.exists():
            raise ValueError('Output file already exists: '+str(path))
    evaluation_years=list(data.evaluation_years)
    print('Checking {} training windows{} ...'.format(len(windows),
          ' and the evaluation sequence' if not args.skip_final_evaluation else ''),flush=True)
    for years in list(windows.values())+([] if args.skip_final_evaluation else [evaluation_years]):
        check_episode_inputs(data,physical,years,days,bins)
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
        release_bounds_scope='current_transition',
        daily_bound_alignment='action_date=t constrains the state after action t; no additional date shift',
        step_logging=dict(enabled=args.episode_log is not None,
            format='compact_csv_v1',
            record_scope='one row per environment day; scalar team metrics over all five reservoirs',
            files={str(seed):str(path.resolve()) for seed,path in daily_paths.items()},
            columns=list(STEP_METRIC_FIELDS),
            episode_step_index='one-based; restarts for each episode',
            daily_reward_discounting='undiscounted; apply gamma**(episode_step-1) for episode totals',
            shortfall_aggregation='physical deficits are summed over the five reservoirs; penalties retain reward weights and means',
            missing_service_values='empty CSV cells; known zero deficits are written as 0',
            float_precision='10 significant digits in CSV only; internal calculations retain full precision',
            output='one CSV per seed; JSONL contains summaries and configuration'),
        evaluation='held-out representative years in configured order; storage carries across boundaries; '
                   'one sequence scored after training, without policy ranking',
        config=config,physical_config=physical,data_hashes=data.hashes,
        source_code_hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})
    print(json.dumps(metadata,indent=2,allow_nan=False),flush=True)
    if args.check_data_only:
        print('Data, episode calendar and observation checks passed; '
              'future trajectory feasibility is not certified. No workers or training started.',flush=True)
        return
    if not args.no_plot:
        subprocess.run([args.plot_python,str(ROOT/'examples/plot_cascade.py'),'--check-backend'],check=True)
    with ExitStack() as outputs:
        stream=None
        daily_writers={}
        if args.episode_log is not None:
            args.episode_log.parent.mkdir(parents=True,exist_ok=True)
            stream=outputs.enter_context(args.episode_log.open('x',encoding='utf-8'))
            daily_writers={seed:StepMetricsWriter(outputs.enter_context(
                path.open('x',encoding='utf-8',newline=''))) for seed,path in daily_paths.items()}
        log_episode(metadata,stream)
        runners=[]
        for seed in config['experiment']['seeds']:
            runner=CascadeReservoirRunner(data,physical,config,seed,episode_log=stream,
                                          step_log=daily_writers.get(seed))
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
          'JSONL summaries and per-seed daily metrics CSVs are saved with --episode-log.',flush=True)
