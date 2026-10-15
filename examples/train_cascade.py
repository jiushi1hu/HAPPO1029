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
from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.reservoir_physics import ReservoirPhysics
from harl.envs.cascade_reservoir.evaluation import reference_summary
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/cascade_reservoir_new")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--beta-concentration-offset", type=float, help="Increase Beta concentration without changing the deterministic action function; 0 adds no concentration")
    parser.add_argument("--beta-action-coordinate", choices=['control','release_fraction'], help="Beta over normalized controls or over their squared release fractions")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--gae-lambda", type=float, help="GAE trace parameter; 1 uses full realized annual returns")
    parser.add_argument("--check-data-only", action="store_true")
    parser.add_argument("--plot-python", default=sys.executable, help="Interpreter with an interactive Matplotlib backend")
    parser.add_argument("--no-plot", action="store_true", help="Run without opening a figure window")
    parser.add_argument("--diagnostics", action="store_true", help="Print training and validation diagnostics without saving files")
    parser.add_argument("--diagnostic-episodes", nargs="+", type=int, default=[0,20,100,150,400])
    parser.add_argument("--diagnostic-stochastic-repeats", type=int, default=3)
    parser.add_argument("--skip-final-test", action="store_true", help="Skip typical-year tests during diagnostic tuning")
    args = parser.parse_args()
    config, physical = load_configuration()
    if args.beta_action_coordinate is not None:
        config['model']['beta_action_coordinate'] = args.beta_action_coordinate
    if args.beta_concentration_offset is not None:
        if not np.isfinite(args.beta_concentration_offset) or args.beta_concentration_offset < 0:
            parser.error("Beta concentration offset must be finite and nonnegative")
        config["model"]["beta_concentration_offset"] = args.beta_concentration_offset
    if args.gae_lambda is not None:
        if not np.isfinite(args.gae_lambda) or not 0. <= args.gae_lambda <= 1.:
            parser.error("GAE lambda must be finite and between 0 and 1")
        config["algo"]["gae_lambda"] = args.gae_lambda
    exp = config["experiment"]
    for key in ("episodes", "eval_interval", "seeds"):
        if getattr(args, key) is not None:
            exp[key] = getattr(args, key)
    if exp["episodes"] < 1 or exp["eval_interval"] < 1 or not exp["seeds"] or len(set(exp["seeds"])) != len(exp["seeds"]):
        parser.error("Positive episodes/interval and nonempty unique seeds are required")
    if args.diagnostic_stochastic_repeats < 1 or any(e < 0 for e in args.diagnostic_episodes):
        parser.error("Positive stochastic repeat count and nonnegative diagnostic episodes required")
    exp.update(diagnostics=args.diagnostics, diagnostic_eval_episodes=args.diagnostic_episodes,
               diagnostic_stochastic_repeats=args.diagnostic_stochastic_repeats)
    data = CascadeData(args.data_dir, physical, exp)
    physics = ReservoirPhysics(data, physical)
    print(json.dumps(dict(period=[str(data.dates[0]), str(data.dates[-1])],
        training_years=list(data.training_years), training_year_count=len(data.training_years),
        validation_years=list(data.validation_years),
        test_years=dict(zip(data.test_labels, data.test_years)),
        episodes_per_seed=exp["episodes"], episode_days="365/366", release_mapping="quadratic",
        reference_energy_MWh=physics.reference_energy,
        beta_concentration_offset=config["model"].get("beta_concentration_offset", 0.),
        beta_action_coordinate=config['model'].get('beta_action_coordinate','control'),
        gae_lambda=config["algo"]["gae_lambda"],
        result_storage="memory only"), indent=2), flush=True)
    if args.check_data_only:
        print("Data checks passed; no training or evaluation was run.")
        return
    # Cache is temporary, and no models, logs or figures are written to the repository.
    with tempfile.TemporaryDirectory(prefix="cascade_happo_plot_") as cache:
        plot_env = dict(os.environ, MPLCONFIGDIR=cache, PYTHONDONTWRITEBYTECODE="1")
        if not args.no_plot:
            subprocess.run([args.plot_python, str(ROOT / "examples/plot_cascade.py"), "--check-backend"],
                           check=True, env=plot_env)
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
        if args.skip_final_test:
            print("Typical-year final tests skipped for this diagnostic run.", flush=True)
        payload = dict(validation_years=list(data.validation_years), test_years=list(data.test_years),
                       test_labels=list(data.test_labels), seeds=exp["seeds"],
                       histories=[r.validation_history for r in runners], tests=tests,
                       references=reference_summary(data, physics, data.validation_years +
                           (() if args.skip_final_test else data.test_years)))
        if not args.no_plot:
            subprocess.run([args.plot_python, str(ROOT / "examples/plot_cascade.py")],
                           input=json.dumps(payload), text=True, check=True, env=plot_env)
    print("Finished. Run results and figure were not saved.", flush=True)


if __name__ == "__main__":
    main()
