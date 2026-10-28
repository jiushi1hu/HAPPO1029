# Five-reservoir HAPPO

Daily operation of the WDD, BHT, XLD, XJB and THR reservoir cascade, based on
PKU-MARL HARL. Five independent categorical actors share a centralized critic.
The environment resolves releases upstream to downstream with zero river
travel delay. HAPPO updates the actors sequentially on a frozen-policy rollout.

## Launch from VS Code

The current standard run uses 1,825-day episodes and six worker processes.
Validation is disabled by default; final testing uses the policy after the
last training update. This command selects 100 discrete actions per reservoir
and an undiscounted objective:

```bash
cd /Users/hujingxin/Documents/Code_python/HARL-main

PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python -B examples/train_cascade.py \
  --collection-mode five_year_parallel --no-validation \
  --episode-days 1825 --parallel-envs 6 --episodes 300 \
  --eval-interval 1 --seeds 1 2 3 4 5 \
  --gamma 1 --action-bins 100 --diagnostics \
  --episode-log runs/five_year_relu_bins100_gamma1_clean_run1.jsonl \
  --plot-python /Library/Frameworks/Python.framework/Versions/3.9/bin/python3
```

Use a new JSONL filename for each run; existing files are never overwritten.
For console-only execution, replace `--plot-python ...` with `--no-plot`.
Launch the file-backed entry point above: spawned workers must be able to
re-import the main module. A Python stdin script is unsuitable.

Check data and full-length physical feasibility without training:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python -B examples/train_cascade.py \
  --collection-mode five_year_parallel --no-validation \
  --episode-days 1825 --parallel-envs 6 --gamma 1 --action-bins 100 \
  --check-data-only
```

`python examples/train_cascade.py --help` lists the supported options. The
repository can be installed into a suitable Python environment with
`python -m pip install -e .`; dependencies are declared in `setup.py`.

## Configuration

`harl/configs/algos_cfgs/happo_cascade.yaml` defines the networks and optimizer.
`happo_cascade_parallel.yaml` overrides collection settings for the standard
parallel mode. `harl/configs/envs_cfgs/cascade_reservoir.yaml` defines physics
and reward assumptions. Command-line settings override these files.

| Setting | Current value |
| --- | --- |
| Collection mode | `five_year_parallel` |
| Episode length / parallel environments | 1,825 days / 6 |
| Episodes / updates per seed | 300 / 50 |
| Collected daily transitions per seed | 547,500 |
| Seeds | 1, 2, 3, 4, 5 |
| Encoder / additional hidden head | `[64, 128, 256]` / 256 |
| Stored hidden widths | `[64, 128, 256, 256]` |
| Hidden activation / layer normalization | ReLU / after each hidden activation |
| Actor / critic learning rate | 0.0003 / 0.0003 |
| Actor / critic epochs per update | 5 / 5 |
| Actor / critic minibatches | 1 / 1 |
| PPO clip / entropy coefficient | 0.2 / 0.001 |
| Maximum gradient norm | 0.5 |
| GAE lambda | 0.95 |
| ValueNorm / shared advantage normalization | Enabled / enabled |
| Actor / joint KL limits | Disabled unless explicitly requested |
| Validation | Disabled by default |
| Terminal storage | Return to episode initial storage |

The YAML defaults remain **200 action bins and gamma=0.99**. The launch command
above explicitly uses **100 bins and gamma=1**. A controlled lambda=1 experiment
can use `--gae-lambda 1`; lambda=1 is not the default. Cleanup of old code does
not change these learning settings.

With seasonal constraints, actor observations have 16 features and the critic
receives 43 state values. The command above constructs:

```text
Each actor: 16 -> 64 -> 128 -> 256 -> 256 -> 100 logits
Critic:     43 -> 64 -> 128 -> 256 -> 256 -> 1 value
```

Each actor and the critic have separate weights. Hidden layers use
`Linear -> ReLU -> LayerNorm`; logits and the final value output are linear.
Without seasonal constraints, observation/state sizes become 11/18.

## Data and episode construction

Source data live in `data/cascade_reservoir_new/` and cover 1965--2024. The
default training pool contains all 55 years outside the five test years:
1998, 1999, 1970, 1997 and 2006. Inflow normalization uses training years only.
Former validation years 2020--2024 are now included in training. Explicit
`--training-years` subsets leave unassigned years unused.

Five-year windows start on January 1 and contain exactly 1,825 consecutive
source days, retaining leap days. The horizon can therefore finish one or two
days before its fifth calendar anniversary. All days must belong to the
training split; the default split provides 39 eligible windows.

Storage, previous releases, discount clocks and return estimation carry across
New Year. Seasonal flood/refill deadlines apply each year. Terminal closure
applies once, at the end of the complete episode. Episode counts must be
divisible by the number of parallel environments.

Final testing concatenates the configured representative test years in their
listed order, carries storage between them, and uses the same episode length.
This is an explicitly constructed representative sequence, rather than a
continuous historical record. `--skip-final-test` reserves test data during
diagnostic runs.

Optional validation remains available with
`--validation-years 2020 2021 2022 2023 2024`. It reserves those years and uses
their continuous sequence for checkpoint selection. It is mutually exclusive
with `--no-validation`. Annual collection remains available through
`--collection-mode annual`; the shared annual runner also provides the base
training implementation used by the parallel runner. A fixed-year diagnostic
uses `--fixed-training-year YEAR` and annual collection.

## Physics and reward

Hard constraints cover double-precision water balance, storage bounds, outlet
capacity, downstream compatibility, seasonal flood/refill bounds and terminal
storage. A remaining-horizon feasibility calculation uses the supplied future
inflows to protect these constraints. It does not optimize generation or enforce
the two soft service requirements.

The default head model is `storage_sensitivity`. It computes head using the
mean of starting and ending forebay elevations, constant efficiency 0.9 and a
supplied offset/slope scenario. This is an assumed sensitivity model, not a
calibrated tailwater model. `--head-model fixed` selects typical constant heads.
Power is limited by turbine flow capacity and installed capacity. Physical
assumptions and source provenance remain in the data-directory documentation.

For discrete index `k` with `K` actions:

```text
u = k / (K - 1)
release = feasible_low + u^2 * (feasible_high - feasible_low)
```

The team reward is normalized daily generation minus the mean relative
guaranteed-output deficit and the mean relative ecological-flow deficit.
Both penalty weights default to 1. Means use the fixed set of reservoirs with
positive requirements. Generation is normalized by 1,624,800 MWh, the installed
capacity energy for one day. Service requirements are recorded in
`reservoir_service_requirements.csv` with their paper source.

Gamma is used consistently in return estimation, actor time weights, episode
accounting and optional checkpoint selection. Gamma=1 gives an undiscounted
generation-minus-penalties objective. Soft penalties express a tradeoff and
do not guarantee zero violations. Physical deficit counts, volumes and energy
shortfalls are reported without discounting. `violation_count=0` in a feasible
evaluation refers to hard constraints; service shortfalls are reported
separately. `--no-reward-penalties` selects a generation-only experiment.

## Execution flow and important files

```text
examples/train_cascade.py
  -> examples/train_cascade_parallel.py
  -> CascadeParallelRunner
  -> six ParallelCascadeEnvs workers
  -> complete [1825, 6, ...] rollout buffers
  -> returns/advantages -> sequential HAPPO actors -> critic update
  -> final-policy tests -> training curves
```

| File or directory | Responsibility |
| --- | --- |
| `examples/train_cascade.py` | CLI, configuration, split resolution, annual mode and parallel routing |
| `examples/train_cascade_parallel.py` | Parallel preflight, experiment orchestration, logs, final testing and plotting |
| `examples/plot_cascade_parallel.py` | Recorded parallel training curves and optional validation curves |
| `examples/plot_cascade.py` | Plot backend checking and annual diagnostic figures |
| `harl/runners/cascade_parallel_runner.py` | Batched process collection and sequence evaluation |
| `harl/runners/cascade_reservoir_runner.py` | Shared network/buffer construction, training loop and annual collection |
| `harl/runners/on_policy_ha_runner.py` | Sequential HAPPO updates and likelihood factors |
| `harl/algorithms/actors/happo.py` | PPO objective, categorical policy updates and optional KL controls |
| `harl/algorithms/critics/v_critic.py` | Value loss, ValueNorm and critic optimization |
| `harl/envs/cascade_reservoir/` | Data, episode views, environment, physics, feasibility, reward and runtime diagnostics |
| `harl/common/` and `harl/models/` | Rollout buffers, normalization and actor/critic networks |

`--diagnostics` records actual parameter changes, categorical entropy/KL,
surrogate changes, Monte Carlo versus GAE fit, service deficits and optimizer
steps. Runtime diagnostics remain part of training. Default curves show
unsmoothed stochastic training returns and generation averaged across the six
collected episodes before that round's update. They are in-sample measurements;
sampled windows also affect their variation.

Models remain in memory; model persistence/resume is not implemented. JSONL
metrics are saved only when `--episode-log` is provided. Plot windows are not
automatically exported. Existing files under `runs/` and `result/` are retained
as historical experiment results.

## Repository cleanup

The test directory and the three standalone annual comparison drivers were
removed from the working repository on 2026-10-28. Generated bytecode caches
and obsolete experiment/test commands were also removed. The active training,
evaluation, diagnostics, plotting, packaging and configuration files remain.
