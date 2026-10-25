"""Annual HAPPO with multi-year training or a fixed-year learning check."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from contextlib import ExitStack

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml
from harl.envs.cascade_reservoir.data_loader import CascadeData, HEAD_MODELS, IDS, resolve_year_split
from harl.envs.cascade_reservoir.cascade_reservoir_env import (
    ACTOR_OBSERVATION_LAYOUT, actor_observation_features)
from harl.envs.cascade_reservoir.reservoir_physics import ReservoirPhysics
from harl.envs.cascade_reservoir.evaluation import (
    reference_summary, reference_metadata, validate_operating_feasibility)
from harl.runners.cascade_reservoir_runner import (
    CascadeReservoirRunner, validate_discount_factor, validate_gae_lambda, validate_seed,
    validate_fixed_training_year)
from harl.envs.cascade_reservoir.reward import reward_settings, REWARD_AGGREGATION


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and key in result else copy.deepcopy(value)
    return result


def load_configuration():
    config_dir = ROOT / "harl/configs"
    def read(rel):
        return yaml.safe_load((config_dir / rel).read_text(encoding="utf-8"))
    return merge(read("algos_cfgs/happo.yaml"), read("algos_cfgs/happo_cascade.yaml")), read("envs_cfgs/cascade_reservoir.yaml")


def reported_references(data, physics, years, action_bins, phase, gamma):
    rows = []
    for year in years:
        print('Computing {} reference for {} ...'.format(phase, year), flush=True)
        row = reference_summary(data, physics, [year], action_bins=action_bins, gamma=gamma)[0]
        print('REFERENCE_{} {}'.format(phase.upper(), json.dumps(row)), flush=True)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/cascade_reservoir_new")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument('--training-years', nargs='+', type=int, metavar='YEAR',
                        help='Explicit training years; unassigned years are unused')
    parser.add_argument('--validation-years', nargs='+', type=int, metavar='YEAR',
                        help='Held-out years whose mean reward selects the checkpoint')
    parser.add_argument('--test-years', nargs='+', type=int, metavar='YEAR',
                        help='Held-out years evaluated only after validation selection')
    parser.add_argument('--fixed-training-year', type=int, metavar='YEAR',
                        help='Train on one training-split year and evaluate it deterministically; '
                             'skip held-out validation, final tests and reference runs')
    parser.add_argument('--action-distribution', choices=['categorical'], help='Categorical discrete indices')
    parser.add_argument('--action-bins', type=int, help='Number of discrete choices per reservoir (default: 200)')
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--gamma", type=float,
                        help="Shared discount factor for training, evaluation and rewards in [0, 1]; overrides YAML")
    parser.add_argument("--gae-lambda", type=float,
                        help="GAE trace parameter; 1 uses full realized returns with the selected gamma")
    parser.add_argument("--head-model", choices=HEAD_MODELS,
                        help="Storage-dependent assumed head scenario, or the original fixed head")
    seasonal = parser.add_mutually_exclusive_group()
    seasonal.add_argument('--seasonal-constraints', dest='seasonal_constraints', action='store_true',
                          help='Enforce supplied seasonal bounds and deadlines (default from YAML: enabled)')
    seasonal.add_argument('--no-seasonal-constraints', dest='seasonal_constraints', action='store_false',
                          help='Use global bounds, 11 local actor inputs, an 18-value critic state and the constant-storage reference')
    parser.set_defaults(seasonal_constraints=None)
    penalties = parser.add_mutually_exclusive_group()
    penalties.add_argument('--reward-penalties',dest='reward_penalties',action='store_true',
                           help='Subtract configured soft-operation penalties from the reward')
    penalties.add_argument('--no-reward-penalties',dest='reward_penalties',action='store_false',
                           help='Use generation-only rewards and model selection')
    parser.set_defaults(reward_penalties=None)
    parser.add_argument("--check-data-only", action="store_true")
    parser.add_argument("--plot-python", default=sys.executable, help="Interpreter with an interactive Matplotlib backend")
    parser.add_argument("--no-plot", action="store_true", help="Run without opening a figure window")
    parser.add_argument('--plot-details', action='store_true', help='Also open reward, training generation, and optimization windows')
    parser.add_argument('--episode-log', type=Path,
                        help='Write episode metrics and fixed-year checkpoints as JSONL to a new file (never overwrite)')
    parser.add_argument("--diagnostics", action="store_true", help="Print training and validation diagnostics without saving files")
    parser.add_argument("--diagnostic-episodes", nargs="+", type=int, default=[0,20,100,150,400])
    parser.add_argument("--diagnostic-stochastic-repeats", type=int, default=3)
    parser.add_argument("--skip-final-test", action="store_true", help="Skip typical-year tests during diagnostic tuning")
    args = parser.parse_args()
    if args.episode_log is not None and args.episode_log.exists():
        parser.error('episode-log must name a new file; existing paths are not overwritten')
    config, physical = load_configuration()
    try:
        config["algo"]["gamma"] = validate_discount_factor(
            config["algo"]["gamma"] if args.gamma is None else args.gamma)
        config["algo"]["gae_lambda"] = validate_gae_lambda(
            config["algo"]["gae_lambda"] if args.gae_lambda is None else args.gae_lambda)
    except ValueError as exc:
        parser.error(str(exc))
    if args.head_model is not None:
        physical["head_model"] = args.head_model
    if args.seasonal_constraints is not None:
        physical['seasonal_constraints'] = args.seasonal_constraints
    if not isinstance(physical.get('seasonal_constraints', False), bool):
        parser.error('seasonal_constraints must be a boolean')
    if not isinstance(physical.get('reward_penalties',{}),dict):
        parser.error('reward_penalties must be a mapping')
    physical['reward_penalties'] = dict(physical.get('reward_penalties',{}))
    if args.reward_penalties is not None:
        physical['reward_penalties']['enabled'] = args.reward_penalties
    elif args.seasonal_constraints is False:
        # The explicit legacy switch restores the generation-only objective.
        physical['reward_penalties']['enabled'] = False
    try:
        penalty_settings = reward_settings(physical)
    except ValueError as exc:
        parser.error(str(exc))
    if args.action_distribution is not None:
        config['model']['action_distribution'] = args.action_distribution
    if args.action_bins is not None:
        if args.action_bins < 2:
            parser.error('action-bins must be >= 2')
        config['model']['action_bins'] = args.action_bins
    if config['model']['action_distribution'] != 'categorical':
        parser.error('Cascade training requires categorical discrete actions')
    exp = config["experiment"]
    if args.fixed_training_year is not None:
        exp['fixed_training_year'] = args.fixed_training_year
    for key in ("episodes", "eval_interval", "seeds"):
        if getattr(args, key) is not None:
            exp[key] = getattr(args, key)
    for key in ('training_years', 'validation_years', 'test_years'):
        if getattr(args, key) is not None:
            exp[key] = getattr(args, key)
    if args.test_years is not None:
        # A custom test year must not inherit an unrelated wet/dry label.
        exp.pop('test_labels', None)
    try:
        split = resolve_year_split(exp)
    except ValueError as exc:
        parser.error(str(exc))
    exp['test_labels'] = split['test_labels']
    try:
        if not isinstance(exp['seeds'], (list, tuple)):
            raise ValueError('Seeds must be a nonempty sequence of integers')
        exp['seeds'] = [validate_seed(seed) for seed in exp['seeds']]
    except ValueError as exc:
        parser.error(str(exc))
    if exp["episodes"] < 1 or exp["eval_interval"] < 1 or not exp["seeds"] or len(set(exp["seeds"])) != len(exp["seeds"]):
        parser.error("Positive episodes/interval and nonempty unique seeds are required")
    if args.diagnostic_stochastic_repeats < 1 or any(e < 0 for e in args.diagnostic_episodes):
        parser.error("Positive stochastic repeat count and nonnegative diagnostic episodes required")
    exp.update(diagnostics=args.diagnostics, diagnostic_eval_episodes=args.diagnostic_episodes,
               diagnostic_stochastic_repeats=args.diagnostic_stochastic_repeats)
    data = CascadeData(args.data_dir, physical, exp)
    try:
        fixed_year = validate_fixed_training_year(exp.get('fixed_training_year'), data)
    except ValueError as exc:
        parser.error(str(exc))
    exp['fixed_training_year'] = fixed_year
    exp['episode_mode'] = 'fixed_training_year' if fixed_year is not None else 'random_training_year'
    physics = ReservoirPhysics(data, physical)
    seasonal_enabled = physical.get('seasonal_constraints', False)
    operation = data.load_seasonal_operation() if seasonal_enabled else None
    comparison = reference_metadata(physical)
    print(json.dumps(dict(period=[str(data.dates[0]), str(data.dates[-1])],
        training_years=list(data.training_years), training_year_count=len(data.training_years),
        episode_mode=exp['episode_mode'], fixed_training_year=fixed_year,
        active_training_years=[fixed_year] if fixed_year is not None else list(data.training_years),
        checkpoint_evaluation=('deterministic evaluation on the fixed training year (in-sample)' if
                               fixed_year is not None else 'held-out validation on {} years'.format(len(data.validation_years))),
        held_out_evaluation_enabled=fixed_year is None,
        validation_years=list(data.validation_years),
        test_years=dict(zip(data.test_labels, data.test_years)),
        episodes_per_seed=exp["episodes"], episode_days="365/366", release_mapping="quadratic",
        reference_energy_MWh=physics.reference_energy,
        head_model=physics.head_model, head_data_status=physics.head_data_status,
        head_time_step_rule=("mean of starting and ending forebay elevations"
                             if physics.head_model == "storage_sensitivity" else "constant typical net head"),
        action_distribution=config['model']['action_distribution'],
        action_bins=config['model']['action_bins'],
        seasonal_constraints=seasonal_enabled,
        observation_size=len(actor_observation_features(seasonal_enabled)),
        actor_observation_size=len(actor_observation_features(seasonal_enabled)),
        actor_observation_layout=ACTOR_OBSERVATION_LAYOUT,
        actor_observation_features=list(actor_observation_features(seasonal_enabled)),
        critic_state_size=43 if seasonal_enabled else 18,
        terminal_storage_mode=physical.get('terminal_storage_mode', 'free'),
        operation_data_status=(None if operation is None else
            {name: operation.key_parameters[name]['value_status'] for name in IDS}),
        gamma=config["algo"]["gamma"],
        gae_lambda=config["algo"]["gae_lambda"],
        reward_penalties=penalty_settings,
        penalty_normalization='mean of excess metres / operating level range over five reservoirs',
        penalty_weight_status='assumed_sensitivity_only',
        evaluation_metric=('gamma-weighted annual penalized return' if penalty_settings['enabled'] else
                           'gamma-weighted annual generation reward'),
        generation_metric='undiscounted annual generation',
        reward_aggregation=REWARD_AGGREGATION,
        result_storage="memory and console; optional episode JSONL",
        episode_log=None if args.episode_log is None else str(args.episode_log.resolve()),
        reward_curve_units='gamma-weighted episode sums; penalties are positive costs',
        penalty_ratio_definition='100 * episode applied penalty / episode Power Reward; null if Power Reward is zero',
        hard_failure_rate_definition='failed episode attempts / attempted episodes, per seed; failures still raise',
        **comparison), indent=2), flush=True)
    selected_years = ([fixed_year] if fixed_year is not None else
                      sorted(set(data.training_years + data.validation_years + data.test_years)))
    print('Checking annual operating feasibility for years {} ...'.format(selected_years), flush=True)
    checked = validate_operating_feasibility(data, physical, years=selected_years)
    print('Operating feasibility passed for {} years (physical bounds and deadlines only).'.format(
        len(checked)), flush=True)
    if args.check_data_only:
        print("Data checks passed; no training or evaluation was run.")
        return
    # Only an explicitly requested episode log persists; figures/models are not exported.
    with tempfile.TemporaryDirectory(prefix="cascade_happo_plot_") as cache, ExitStack() as stack:
        plot_env = dict(os.environ, MPLCONFIGDIR=cache, PYTHONDONTWRITEBYTECODE="1")
        if not args.no_plot:
            subprocess.run([args.plot_python, str(ROOT / "examples/plot_cascade.py"), "--check-backend"],
                           check=True, env=plot_env)
        episode_log = None
        if args.episode_log is not None:
            args.episode_log.parent.mkdir(parents=True, exist_ok=True)
            episode_log = stack.enter_context(args.episode_log.open('x', encoding='utf-8'))
        references = []
        if fixed_year is None:
            references = reported_references(data, physics, data.validation_years,
                                             config['model']['action_bins'], 'validation', config['algo']['gamma'])
            print('Validation reference mean = {:.3f} TWh/year ({})'.format(
                np.mean([r['energy_TWh'] for r in references]), comparison['reference_label']), flush=True)
        else:
            print('Fixed-year learning check: training and deterministic evaluation use {}. '
                  'Held-out validation, final tests and reference runs are skipped.'.format(fixed_year), flush=True)
        runners = []
        for seed in exp["seeds"]:
            runner = CascadeReservoirRunner(data, physical, config, seed,
                **({'episode_log': episode_log} if episode_log is not None else {}))
            runner.run(exp["episodes"], exp["eval_interval"])
            runners.append(runner)
        # Test years are used only after all validation-based selections finish.
        skip_tests = args.skip_final_test or fixed_year is not None
        tests = [] if skip_tests else [row for runner in runners for row in runner.final_test()]
        for year, label in ([] if skip_tests else zip(data.test_years, data.test_labels)):
            values = np.array([r["energy_TWh"] for r in tests if r["year"] == year])
            print("test {} ({}) mean={:.3f} TWh seed_SD={:.3f} TWh".format(
                year, label, values.mean(), values.std(ddof=1) if len(values) > 1 else 0.), flush=True)
            selected = [r for r in tests if r['year']==year]
            print('test {} mean_reward={:.6f} mean_penalty={:.6f} gamma={}'.format(year,
                np.mean([r['episode_reward'] for r in selected]),
                np.mean([r['episode_penalty'] for r in selected]),config['algo']['gamma']),flush=True)
        if skip_tests:
            print("Typical-year final tests skipped for this diagnostic run.", flush=True)
        else:
            references.extend(reported_references(data, physics, data.test_years,
                                                  config['model']['action_bins'], 'test', config['algo']['gamma']))
        payload = dict(gamma=config["algo"]["gamma"], gae_lambda=config["algo"]["gae_lambda"],
                       reward_aggregation=REWARD_AGGREGATION,
                       fixed_training_year=fixed_year,
                       fixed_year_histories=([r.fixed_year_history for r in runners] if fixed_year is not None else []),
                       validation_years=list(data.validation_years), test_years=list(data.test_years),
                       test_labels=list(data.test_labels), seeds=exp["seeds"],
                       histories=[r.validation_history for r in runners], tests=tests,
                       episode_histories=[r.episode_history for r in runners],
                       references=references, seasonal_constraints=seasonal_enabled,
                       action_bins=config['model']['action_bins'], head_model=physics.head_model,
                       head_data_status=physics.head_data_status,
                       terminal_storage_mode=physical.get('terminal_storage_mode', 'free'),
                       reward_penalties=penalty_settings, **comparison)
        if args.plot_details:
            payload.update(training_histories=[r.training_history for r in runners],
                           reference_energy_MWh=physics.reference_energy)
        if not args.no_plot:
            subprocess.run([args.plot_python, str(ROOT / "examples/plot_cascade.py")] +
                           (['--plot-details'] if args.plot_details else []),
                           input=json.dumps(payload), text=True, check=True, env=plot_env)
    print('Finished. Models and figures were not saved.' +
          (' Episode metrics saved to {}.'.format(args.episode_log.resolve()) if args.episode_log is not None
           else ' Episode metrics remain in memory and console.'), flush=True)


if __name__ == "__main__":
    main()
