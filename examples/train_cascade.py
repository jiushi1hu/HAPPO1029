"""Annual HAPPO on 50 real inflow years, with validation and typical-year tests."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml
from harl.envs.cascade_reservoir.data_loader import CascadeData, HEAD_MODELS, IDS
from harl.envs.cascade_reservoir.reservoir_physics import ReservoirPhysics
from harl.envs.cascade_reservoir.evaluation import (
    reference_summary, reference_metadata, validate_operating_feasibility)
from harl.runners.cascade_reservoir_runner import (
    CascadeReservoirRunner, validate_discount_factor, validate_gae_lambda, validate_seed)
from harl.envs.cascade_reservoir.reward import reward_settings


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


def reported_references(data, physics, years, action_bins, phase):
    rows = []
    for year in years:
        print('Computing {} reference for {} ...'.format(phase, year), flush=True)
        row = reference_summary(data, physics, [year], action_bins=action_bins)[0]
        print('REFERENCE_{} {}'.format(phase.upper(), json.dumps(row)), flush=True)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/cascade_reservoir_new")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument('--action-distribution', choices=['categorical'], help='Categorical discrete indices')
    parser.add_argument('--action-bins', type=int, help='Number of discrete choices per reservoir (default: 200)')
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--gamma", type=float, help="Training discount factor in [0, 1]; overrides YAML")
    parser.add_argument("--gae-lambda", type=float,
                        help="GAE trace parameter; 1 uses full realized returns with the selected gamma")
    parser.add_argument("--head-model", choices=HEAD_MODELS,
                        help="Storage-dependent assumed head scenario, or the original fixed head")
    seasonal = parser.add_mutually_exclusive_group()
    seasonal.add_argument('--seasonal-constraints', dest='seasonal_constraints', action='store_true',
                          help='Enforce supplied seasonal bounds and deadlines (default from YAML: enabled)')
    seasonal.add_argument('--no-seasonal-constraints', dest='seasonal_constraints', action='store_false',
                          help='Use legacy global bounds, 18 observations and the constant-storage reference')
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
    parser.add_argument("--diagnostics", action="store_true", help="Print training and validation diagnostics without saving files")
    parser.add_argument("--diagnostic-episodes", nargs="+", type=int, default=[0,20,100,150,400])
    parser.add_argument("--diagnostic-stochastic-repeats", type=int, default=3)
    parser.add_argument("--skip-final-test", action="store_true", help="Skip typical-year tests during diagnostic tuning")
    args = parser.parse_args()
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
    for key in ("episodes", "eval_interval", "seeds"):
        if getattr(args, key) is not None:
            exp[key] = getattr(args, key)
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
    physics = ReservoirPhysics(data, physical)
    seasonal_enabled = physical.get('seasonal_constraints', False)
    operation = data.load_seasonal_operation() if seasonal_enabled else None
    comparison = reference_metadata(physical)
    print(json.dumps(dict(period=[str(data.dates[0]), str(data.dates[-1])],
        training_years=list(data.training_years), training_year_count=len(data.training_years),
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
        observation_size=43 if seasonal_enabled else 18,
        terminal_storage_mode=physical.get('terminal_storage_mode', 'free'),
        operation_data_status=(None if operation is None else
            {name: operation.key_parameters[name]['value_status'] for name in IDS}),
        gamma=config["algo"]["gamma"],
        gae_lambda=config["algo"]["gae_lambda"],
        reward_penalties=penalty_settings,
        penalty_normalization='mean of excess metres / operating level range over five reservoirs',
        penalty_weight_status='assumed_sensitivity_only',
        evaluation_metric=('undiscounted annual penalized return' if penalty_settings['enabled'] else
                           'undiscounted annual generation'),
        generation_metric='undiscounted annual generation',
        result_storage="memory only", **comparison), indent=2), flush=True)
    print('Checking annual operating feasibility for all supplied years ...', flush=True)
    checked = validate_operating_feasibility(data, physical)
    print('Operating feasibility passed for {} years (physical bounds and deadlines only).'.format(
        len(checked)), flush=True)
    if args.check_data_only:
        print("Data checks passed; no training or evaluation was run.")
        return
    # Cache is temporary, and no models, logs or figures are written to the repository.
    with tempfile.TemporaryDirectory(prefix="cascade_happo_plot_") as cache:
        plot_env = dict(os.environ, MPLCONFIGDIR=cache, PYTHONDONTWRITEBYTECODE="1")
        if not args.no_plot:
            subprocess.run([args.plot_python, str(ROOT / "examples/plot_cascade.py"), "--check-backend"],
                           check=True, env=plot_env)
        references = reported_references(data, physics, data.validation_years,
                                         config['model']['action_bins'], 'validation')
        print('Validation reference mean = {:.3f} TWh/year ({})'.format(
            np.mean([r['energy_TWh'] for r in references]), comparison['reference_label']), flush=True)
        runners = []
        for seed in exp["seeds"]:
            runner = CascadeReservoirRunner(data, physical, config, seed)
            runner.run(exp["episodes"], exp["eval_interval"])
            runners.append(runner)
        # Test years are used only after all validation-based selections finish.
        tests = [] if args.skip_final_test else [row for runner in runners for row in runner.final_test()]
        for year, label in ([] if args.skip_final_test else zip(data.test_years, data.test_labels)):
            values = np.array([r["energy_TWh"] for r in tests if r["year"] == year])
            print("test {} ({}) mean={:.3f} TWh seed_SD={:.3f} TWh".format(
                year, label, values.mean(), values.std(ddof=1) if len(values) > 1 else 0.), flush=True)
            if penalty_settings['enabled']:
                selected = [r for r in tests if r['year']==year]
                print('test {} mean_reward={:.6f} mean_penalty={:.6f}'.format(year,
                    np.mean([r['episode_reward'] for r in selected]),
                    np.mean([r['episode_penalty'] for r in selected])),flush=True)
        if args.skip_final_test:
            print("Typical-year final tests skipped for this diagnostic run.", flush=True)
        else:
            references.extend(reported_references(data, physics, data.test_years,
                                                  config['model']['action_bins'], 'test'))
        payload = dict(gamma=config["algo"]["gamma"], gae_lambda=config["algo"]["gae_lambda"],
                       validation_years=list(data.validation_years), test_years=list(data.test_years),
                       test_labels=list(data.test_labels), seeds=exp["seeds"],
                       histories=[r.validation_history for r in runners], tests=tests,
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
    print("Finished. Run results and figure were not saved.", flush=True)


if __name__ == "__main__":
    main()
