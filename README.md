# Five-reservoir HAPPO

Daily operation of the WDD, BHT, XLD, XJB and THR reservoir cascade, based on
PKU-MARL HARL. Five independent categorical actors use one centralized critic.
The environment resolves releases upstream to downstream with zero travel delay.

## Current experiment

- Data cover 1965--2024: **55 training years and five evaluation years**.
- Evaluation years, in sequence order: **1998, 1999, 1970, 1997, 2006**.
- Training years are the full complement; inflow scaling uses training data only.
- Each episode contains **1,825 daily decisions**, with **six parallel workers**.
- Each reservoir has **100 discrete actions** by default.
- The final trained policy is evaluated after all training seeds finish.

The standard budget is 300 completed episodes per seed: 50 HAPPO updates and
547,500 daily environment transitions. Each update collects six complete
episodes under one frozen policy, giving buffers of shape `[1825, 6, ...]`.
Episode counts must be divisible by the number of workers.

## Launch from VS Code

```bash
cd /Users/hujingxin/Documents/Code_python/HARL-main

PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python -B examples/train_cascade.py \
  --episodes 300 --seeds 1 2 3 4 5 --gamma 1 --diagnostics \
  --episode-log runs/five_year_100actions_gamma1_run1.jsonl \
  --plot-python /Library/Frameworks/Python.framework/Versions/3.9/bin/python3
```

The episode horizon, worker count and action count already default to
1,825 / 6 / 100. They can be stated explicitly with
`--episode-days 1825 --parallel-envs 6 --action-bins 100`.
Use a new log filename for every run; existing files are never overwritten.
For console-only execution, replace `--plot-python ...` with `--no-plot`.

Launch the file-backed entry point above so spawned workers can re-import
the main module. Model persistence/resume is not implemented; start a fresh
run to use changed source or settings. Metrics are saved only when an
`--episode-log` path is supplied; plot windows are not automatically exported.

Check all training windows and the evaluation sequence without training:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python -B examples/train_cascade.py \
  --gamma 1 --check-data-only
```

`--skip-final-evaluation` reserves held-out data during tuning.
`--log-interval N` controls training diagnostic reporting in update rounds.
`--evaluation-years` accepts exactly five unique years and uses the remaining
55 for training. `--help` describes the supported flags. The CLI exposes only
the current multi-year training pipeline.

## Configuration and networks

`harl/configs/algos_cfgs/happo_cascade.yaml` contains the complete training,
network, optimizer and split settings. There is no separate collection overlay.
`harl/configs/envs_cfgs/cascade_reservoir.yaml` contains physical assumptions and
reward settings. CLI arguments override their corresponding YAML values.

| Setting | Default |
| --- | --- |
| Episode horizon / parallel environments | 1,825 days / 6 |
| Discrete actions per reservoir | 100 |
| Episodes per seed / seeds | 300 / 1, 2, 3, 4, 5 |
| Encoder / additional hidden head | `[64, 128, 256]` / 256 |
| Stored hidden widths / activation | `[64, 128, 256, 256]` / ReLU |
| Actor / critic learning rate | 0.0003 / 0.0003 |
| Actor / critic optimization epochs | 5 / 5 |
| Actor / critic minibatches per epoch | 1 / 1 |
| Gamma / GAE lambda | 0.99 / 0.95 |
| PPO clip / entropy coefficient | 0.2 / 0.001 |
| Maximum gradient norm | 0.5 |
| ValueNorm / shared advantage normalization | Enabled / enabled |
| Actor / joint KL limits | Disabled unless requested |
| Terminal storage | Return to the episode's initial storage |

The launch command explicitly selects gamma=1 for undiscounted five-year net
return. A separate lambda experiment can use `--gae-lambda 1`; it is not the
default. Episode duration, split terminology and action defaults were updated
without changing reward coefficients or optimizer hyperparameters.

With seasonal constraints, each actor observes 16 local/neighbor values and
the critic receives 43 centralized state values:

```text
Each actor: 16 -> 64 -> 128 -> 256 -> 256 -> 100 categorical logits
Critic:     43 -> 64 -> 128 -> 256 -> 256 -> 1 value
```

Each network has separate weights. Hidden layers use
`Linear -> ReLU -> LayerNorm`; logits and the final value remain linear.
Without seasonal constraints, input dimensions become 11/18.

## Episode calendar and evaluation

Training windows start on January 1 and contain exactly 1,825 consecutive
source days. Leap days are retained, so the horizon can finish one or two
days before its fifth calendar anniversary. Every executed day must belong
to training data; the default split provides 39 eligible windows.

Storage, previous releases, GAE and discount clocks carry across calendar
boundaries. Flood-control and refill deadlines still apply in each source
year. Terminal closure applies once at the end of the full episode. Time-to-go
features are scaled by the complete episode horizon.

Evaluation joins the five representative years in their configured order,
carries storage between them, and uses the same episode horizon. Their forcing
and seasonal rules are retained. This is a constructed representative sequence,
rather than a continuous historical record. The evaluator scores the final
policy without ranking intermediate policies.

`--episode-days` also supports explicitly shortened execution checks. Their
actual dates and horizon are recorded; their metrics do not describe a
full-length experiment.

## Physics, actions and reward

Raw data and source notes live in `data/cascade_reservoir_new/`. Hard constraints
cover water balance, storage, outlet capacity, downstream compatibility,
seasonal deadlines and terminal storage. Remaining-horizon feasibility uses
the supplied future inflows. It does not optimize energy or enforce the two
soft service requirements.

The default `storage_sensitivity` head model uses mean starting/ending forebay
elevation, constant efficiency 0.9, and supplied slope/offset assumptions.
This is a sensitivity scenario, not a calibrated tailwater model. Turbine flow
and installed capacity limit generation. `--head-model fixed` uses typical heads.

For index `k` among `K=100` actions:

```text
u = k / (K - 1)
release = feasible_low + u^2 * (feasible_high - feasible_low)
```

Daily team reward is normalized generation minus mean relative guaranteed-output
and ecological-flow deficits, each with weight 1. Generation is normalized by
1,624,800 MWh. Requirements and their source are retained in
`reservoir_service_requirements.csv`. `--no-reward-penalties` selects a separate
generation-only objective.

Gamma applies consistently to return estimation, actor time weights and episode
reward accounting. Physical violation counts, deficit volumes and deficit
energy remain undiscounted. Episode generation and return are totals over the
whole horizon. Soft costs express a tradeoff; hard feasibility does not imply
zero service shortfalls.

## Source and output structure

| File or directory | Responsibility |
| --- | --- |
| `examples/train_cascade.py` | CLI, complete configuration and 55/5 split |
| `examples/train_cascade_parallel.py` | Full-window preflight, seed orchestration, logging and final evaluation |
| `examples/plot_cascade.py` | Interactive backend availability check |
| `examples/plot_cascade_parallel.py` | Unsmoothed training return and generation curves |
| `harl/runners/cascade_reservoir_runner.py` | Shared actor/critic construction, parallel collection, update loop and final evaluation |
| `harl/runners/on_policy_ha_runner.py` | Sequential HAPPO updates and likelihood factors |
| `harl/algorithms/` | Actor and critic objectives, optimizer steps and optional KL controls |
| `harl/envs/cascade_reservoir/` | Data, episode views, workers, physics, feasibility, reward and training diagnostics |
| `harl/common/` and `harl/models/` | Buffers, normalization and feed-forward networks |

`--diagnostics` reports parameter changes, entropy/KL, surrogate changes and
Monte Carlo/GAE fit on collected training trajectories. Curves show recorded
stochastic returns and generation, averaged across the six complete episodes
collected before an update. They begin after collection and have no fabricated
episode-zero point. Sampled inflow windows affect their variation.

New JSONL records use `run_configuration`, `training_episode`, `training_update`,
`training_diagnostics`, `final_evaluation_sequence` and `run_complete`.
Per-episode records include source years, dates and the actual horizon;
generation units refer to the full episode. Historical files in `runs/` and
`result/` retain the settings and schemas of the runs that produced them.

Dependencies are declared in `setup.py`; install into a suitable Python
environment with `python -m pip install -e .`.
