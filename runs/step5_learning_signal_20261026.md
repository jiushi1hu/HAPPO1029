# Step 5: learning-signal diagnosis and controlled experiments

Implemented training-only signal diagnostics and a one-factor comparison driver
for GAE lambda or annual rollout batch size. The normal training defaults remain
unchanged: lambda 0.95, six annual trajectories per update, five actor epochs and
five critic epochs. No automatic update rejection, learning-rate adjustment or
held-out acceptance rule was added in this step.

## Why this addresses the next question

The earlier checks showed modest policy KL, and the joint cap did not ensure a
better deterministic controller. A tighter policy-change limit cannot establish
whether the direction supplied by the return estimator is useful. This step
therefore exposes the training signal before changing more optimizer settings.

The existing diagnostics already measured critic fit to realized Monte Carlo
returns and GAE targets. The new report adds frozen target/advantage comparisons,
per-trajectory evidence, actor objective subsets, and annual-year diversity. It
provides a controlled way to examine alternatives before changing defaults.

## Diagnostic definitions

`before_update()` now freezes the stored targets as well as denormalized old
critic values. `learning_signal_report()` measures:

```
MC_t = reward_t + gamma * terminal_mask_(t+1) * MC_(t+1)
raw_GAE_advantage_t = frozen_target_t - frozen_old_value_t
realized_MC_advantage_t = MC_t - frozen_old_value_t
target_gap_t = frozen_target_t - MC_t
```

The report includes target-gap RMSE/bias and explained variance; population
standard deviation, percentiles and ranges of raw advantages; and sign
disagreement where both paired advantages are nonzero. Its global row summaries
are unweighted descriptions of the recorded daily rows. Each actor also gets
statistics on positive-objective rows, a discount-weighted normalized-advantage
mean, the objective weight sum, and the corresponding sign disagreement.
Shared or actor-specific normalization follows the actual training arithmetic.
Forced and zero-weight objectives have empty statistics with JSON null values.

Per-trajectory results include year, length, discounted return, initial target
and old value, MC critic fit before/after training, target gaps and advantage
distributions. The post-training values use the updated ValueNorm statistics;
the old predictions and targets retain their original reward-unit interpretation.
These are in-sample fit measurements, not an independent critic validation.

Batch summaries distinguish annual trajectories from daily rows, record year
frequencies, and report inverse frequency concentration as descriptive year
diversity. Daily rows are temporally correlated. No daily standard error or
independent-sample count is inferred from their number. Evaluation-repeat
uncertainty and training-seed variation remain separate.

MC is the return realized by the sampled trajectory, not the true expected value
or advantage. GAE/MC sign disagreement does not prove that one estimate is
correct. The diagnostic requires true terminal trajectories, no truncation, and
consistent episode boundaries. `use_gae`/`target_estimator` identify ordinary
return targets if GAE is disabled; the historical target-gap field name remains
`gae_target_vs_mc`.

With complete annual termination and lambda one, future-value bootstrap terms
cancel from the targets up to floating-point error. The old critic baseline
still appears in the advantages. Longer-horizon sampled targets can also have
more variability, so lambda one is an experiment rather than an automatic fix.

## Controlled comparison implementation

`examples/compare_cascade_learning_signal.py` supports exactly one intervention:

| Intervention | Baseline | Candidate | Budget interpretation |
|---|---|---|---|
| `gae_lambda` | Configured lambda | Explicit candidate, default 1 | Same annual episodes, batch size and update opportunities |
| `rollout_episodes` | Configured annual batch | Explicit larger batch, default 12 | Same annual episodes; update counts and maximum optimizer opportunities can differ |

Both arms start from identical learning state and receive the same explicit
sampled-year schedule. Networks, gamma, clipping, learning rates, actor/critic
epoch limits, rewards, physical settings and KL protections are common. Partial
final batches are retained. A larger batch changes both frozen-policy collection
and update frequency; it is not a pure variance comparison with identical
optimizer budgets. Actual retained optimizer steps are logged, since KL guards
can change them.

The driver uses a narrow `TrainingSignalRunner` subclass that evaluates current
policies only at initial/final endpoints on the selected training years. It
does not invoke validation selection, final tests or reference controllers.
Evaluation preserves RNG and actor modes and checks the full learning-state
fingerprint. Final sampled differences use paired action-repeat means on a
common per-year seed grid. Seed-level summaries use the paired estimated
differences across trained seeds rather than pooled daily rows or action draws.
Single-repeat and single-seed standard errors are null at the respective level.

Configuration, source/data hashes, year schedules, full training diagnostics,
optimizer counts and endpoint reports are written to a new JSONL file. Existing
outputs are never overwritten. Initial-state mismatch, invalid interventions,
unmatched evaluation grids, extra changed settings or divergent year schedules
are rejected.

## Real two-year lambda check

One bounded annual comparison used training seed 2, years 2012 and 2013, a common
schedule `[2012, 2013]`, two annual episodes, a batch of two, and one update in
each arm. The leap year contributes 366 rows and the common year 365: **731
daily rows, but only two annual trajectories and two distinct years**.

The arms used 50 action bins, the existing 128/128 networks, gamma 0.99, the
default learning rates and five actor/critic epochs, seasonal constraints,
storage-dependent head, service penalties and annual storage closure. All KL
protections were disabled equally in this signal experiment. Lambda alone
changed from 0.95 to 1. Endpoint evaluation uses two sampled repeats per year
and separate deterministic evaluation on the same training-year set.

The first collection is under identical initial policies in both arms. Frozen
old critic fit and realized trajectory returns agree exactly. The initial
realized discounted returns are approximately -53.1801 for 2012 and -39.5363 for
2013. Under lambda 0.95, their initial GAE targets were only -2.09238 and
-2.41941, respectively, with old initial values near 0.0363 and 0.0352.

| Training-batch measurement | Lambda 0.95 | Lambda 1 |
|---|---:|---:|
| Target versus MC RMSE, reward units | 32.9461889 | 0.0000215083 |
| Target minus MC mean | +27.8412745 | -0.0000196975 |
| Raw advantage sign disagreement with MC | 28.1806% | 0% |
| Raw advantage population SD | 10.0306427 | 22.4762033 |
| Critic versus MC RMSE before training | 42.3430599 | 42.3430599 |
| Critic versus MC RMSE after training | 35.0367335 | 22.8279495 |
| Critic versus MC explained variance after training | -0.0269759 | -0.0176096 |

Lambda one makes the targets agree with realized reward-to-go within float32
accumulation error and increases raw advantage spread by about 2.24 times.
The lower MC RMSE after critic training does not mean the critic has learned the
daily variation well: explained variance remains slightly negative in both arms
after this single update. Updated ValueNorm also contributes to the prediction
shift. These observations support examining the signal; they do not establish
that lambda one is the better training setting.

Per-actor mean controllable-state KL after the update was approximately
0.00423–0.00574 for lambda 0.95 and 0.00419–0.00599 for lambda one. Thus the target
and advantage differences occurred while measured policy movement remained
modest. They are not evidence of KL explosion.

Endpoint results are means across the same two training years:

| Evaluation, discounted net return | Common initial policy | Final lambda 0.95 | Final lambda 1 |
|---|---:|---:|---:|
| Deterministic categorical mode | -4.1356548 | -8.4093378 | +4.8265494 |
| Sampled policy, two-repeat mean | -44.9371827 | -43.6131508 | -43.6224351 |

The final candidate-minus-baseline deterministic difference was **+13.2358872**.
The paired sampled differences were **[-0.5423388, +0.5237701]**, with mean
**-0.0092844** and action-sampling standard error **0.5330544**. The two repeats
provide no clear sampled-return improvement. This standard error is conditional
on these evaluation years and one training seed; uncertainty across trained
seeds cannot be estimated here. All endpoint evaluations passed the learning
state preservation check. Both arms retained five optimizer steps per actor,
with matching episode/update/day budgets and identical initial fingerprints.

The result demonstrates the intended target change and the distinction between
sampled and deterministic evaluation. It does not establish that lambda one is
the better default, a long-run convergence improvement or a held-out benefit.
The larger-batch alternative was verified through integration fixtures rather
than a second full annual performance trial in this step.

Reproduce the bounded annual check with a new output path:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python -B examples/compare_cascade_learning_signal.py \
  --intervention gae_lambda --episodes 2 --rollout-episodes 2 \
  --training-years 2012 2013 --seeds 2 --evaluation-repeats 2 \
  --output runs/step5_lambda_2012_2013_seed2_new.jsonl
```

For a broader comparison, increase annual episodes, training years, evaluation
repeats and trained seeds while changing only the named intervention. The README
includes separate lambda and batch-size commands. Validation remains available
for subsequent experiment comparison/checkpoint selection in the normal runner;
test years remain reserved for final evaluation.

## Files and verification

| File | Responsibility |
|---|---|
| `harl/envs/cascade_reservoir/learning_signal.py` | Frozen-signal arithmetic, per-actor/per-trajectory summaries, boundary checks and diversity |
| `harl/envs/cascade_reservoir/diagnostics.py` | Freeze targets, include signal reports, console summaries, training-year evaluation scope |
| `examples/compare_cascade_learning_signal.py` | One-factor paired experiments, explicit schedules, training-only endpoints and provenance |
| `examples/train_cascade.py` | Expanded diagnostics help text |
| `tests/test_learning_signal_diagnostics.py` | Eleven signal, normalization, terminal, ValueNorm, scope and state-preservation tests |
| `tests/test_cascade_learning_comparison.py` | Ten pairing, uncertainty, CLI, endpoint, schedule and budget tests |
| `README.md` | Definitions, interpretation and commands |

**310 tests passed:** 289 core tests with the training interpreter and 21 plotting
tests with the plotting interpreter. Focused tests verify exact preservation of
parameters, Adam state, ValueNorm, gradients, buffers and RNG when diagnostics
are enabled; lambda-one targets across internal annual boundaries with ValueNorm;
and rejection of incomplete or misaligned diagnostics. The larger-batch fixture
retains five complete episodes at batch sizes two and four, producing three and
two updates with the final partial episode retained. `git diff --check` passed.

The annual run records and delivered-source hashes are kept separately. Later
hardening adds invalid-input checks and target-estimator labels without changing
the tested training calculations. Historical step artifacts and the pre-existing
long training log are preserved.

Artifacts:

- [Raw paired annual records](step5_lambda_2012_2013_seed2_20261026.jsonl)
- [Machine-readable verification and delivered-source hashes](step5_learning_signal_summary_20261026.json)
- [Console diagnostics](step5_lambda_2012_2013_seed2_20261026.console.log)
