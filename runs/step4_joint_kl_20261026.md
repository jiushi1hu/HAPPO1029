# Step 4: cumulative KL across sequential HAPPO actors

Implemented an optional common-rollout joint command-policy KL cap. Enable it
with `--joint-target-kl LIMIT --target-kl LOCAL_LIMIT --kl-rollback`. Both limits
must be finite and positive. The default `algo.joint_target_kl: null` preserves
the existing training behavior. The new limit supplements the step-3 local
guard and uses the same per-candidate actor/Adam/gradient rollback.

## What is measured

For each recorded joint observation row, the five independent actors define a
factorized categorical command policy. With every old distribution frozen at
the beginning of the update round:

```
P_old(a | row) = product_i p_old_i(a_i | observation_i)
P_new(a | row) = product_i p_new_i(a_i | observation_i)

KL(P_old || P_new) = sum_i KL(p_old_i || p_new_i)
joint_mean_KL = sum_i mean_over_common_rows(KL(p_old_i || p_new_i))
```

KL uses natural logarithms. Every component uses the same rollout rows, with
uniform averaging and no discount or controllability weights. Stored available
action masks are retained. A nonnegative clamp on each actor's computed mean
suppresses floating-point roundoff; the identity holds up to that roundoff.

The previous local `policy_kl` averages over controllable decisions, and different
reservoirs can have different controllable rows. Adding those differently masked
means would not yield the joint mean above. Consequently the implementation
keeps local KL and the common-row actor contribution as separate metrics.

This is the raw categorical command distribution, before the feasibility shield
maps commands into releases. A changed command distribution on a forced row
still contributes even when that row's executed release is unaffected. The
measurement can therefore be conservative for physical control. It is not a
direct measurement of executed-release KL.

## Acceptance and execution flow

1. The sequential runner constructs `JointKLBudget` once per collected batch,
   before actor optimization. It freezes all five distributions and validates
   matching rollout grids and episode-boundary masks, categorical actions and
   independent actor parameters. Cascade collection supplies aligned rows from
   the same trajectories.
2. Each actor receives a guard tied to that actor and its actual buffer. Earlier
   actors' retained contributions determine its remaining allowance. The
   current actor's reference remains the round-start policy across all epochs
   and minibatches.
3. After each effective optimizer candidate, HAPPO checks finite state, local KL,
   and the candidate joint total. That total replaces the current actor's
   previous contribution; it does not accumulate all its successive KL values.
4. A candidate exceeding either limit is restored completely, including Adam
   moments/counters and gradients, and the rest of that actor's optimization is
   stopped. A rejected candidate consumes no joint budget. Earlier retained
   updates stay in place. Measurement exceptions restore the candidate before
   propagating the error.
5. Later actors can still attempt a smaller step within the remaining budget.
   HAPPO recomputes its likelihood factors from retained actor weights. The
   centralized critic trains afterward as usual.

A later candidate can reduce the current actor's own KL and release some
allowance. Earlier actors are not revisited. The next collected batch starts a
new budget with new references. The existing random actor order is preserved;
this is not equal budget allocation, and earlier actors can consume most of the
allowance. There are no automatic retries or learning-rate reductions.

When disabled, the joint mechanism adds no distribution evaluations, snapshots
or random draws. A nonbinding enabled guard also preserves weights, Adam,
gradients, existing training metrics and RNG exactly in the focused tests.
Actual rejections can change subsequent RNG consumption by skipping remaining
minibatches or epochs, as in step 3.

## Annual correctness stress check

The actual 2013 cascade environment completed one annual training episode and
one update with seed 1, 50 bins, the existing 128/128 networks and learning rates,
five maximum actor epochs, seasonal constraints, variable head, service penalties
and annual storage closure. Local KL was limited to `0.01`; the deliberately
binding joint limit was `0.001`. This is a correctness stress test, not a tuned
threshold or a controlled performance comparison.

Update order and results, with unmasked actor contributions shown:

| Actor, in update order | Allowance at entry | Retained actor KL | Joint KL after actor | Retained steps | Rejected steps |
|---|---:|---:|---:|---:|---:|
| THR | 0.001000000 | 0.000581130 | 0.000581130 | 2 | 1 |
| XJB | 0.000418870 | 0.000132848 | 0.000713978 | 1 | 1 |
| WDD | 0.000286022 | 0.000176725 | 0.000890703 | 1 | 1 |
| XLD | 0.000109297 | 0 | 0.000890703 | 0 | 1 |
| BHT | 0.000109297 | 0 | 0.000890703 | 0 | 1 |

There were nine effective candidates: four retained and five restored. The final
joint KL was **0.0008907030278351158**, below **0.001**, with
**0.00010929697216488423** remaining. All five rejections crossed the joint cap
while their local candidate KLs remained below `0.01`. The retained components
sum exactly to the recorded joint total using the implementation's summation.

BHT and XLD rejected their first candidates and had exactly zero parameter
change, zero mode changes and likelihood ratios equal to one. The other three
actors retained their accepted changes. The critic still updated, with parameter
change L2 norm approximately `0.2062943`.

Small command KL did **not** ensure improved deterministic performance:

| Evaluation | Before update | After update | Change |
|---|---:|---:|---:|
| Deterministic discounted net return | -24.6779268 | -35.6786822 | -11.0007554 |
| One matched sampled discounted net return | -37.2532844 | -36.9056994 | +0.3475850 |
| Deterministic generation, TWh | 228.513009 | 217.351207 | -11.161802 |
| One matched sampled generation, TWh | 196.097046 | 196.633267 | +0.536221 |

The sampled evaluation uses the same seed before and after, but one repeat has
no estimated sampling standard error. Its positive change does not establish
an improvement in expected return. Near-uniform categorical distributions can
change their modal action substantially under small KL changes; observed mode
changes among updated actors were approximately 49.6%, 83.1% and 65.6% on their
controllable rows. This test establishes enforcement of the joint cap, not
monotonic improvement, long-run convergence or held-out superiority.

Reproduction:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python -B examples/train_cascade.py \
  --fixed-training-year 2013 --episodes 1 --rollout-episodes 1 --eval-interval 1 \
  --seeds 1 --action-bins 50 --ppo-epochs 5 --target-kl 0.01 --kl-rollback \
  --joint-target-kl 0.001 --diagnostics --diagnostic-episodes 0 1 \
  --diagnostic-stochastic-repeats 1 --no-plot \
  --episode-log runs/step4_joint_kl_stress_2013_seed1_new.jsonl
```

## Files and diagnostics

| File | Change |
|---|---|
| `harl/algorithms/actors/joint_kl.py` | Frozen common-row references, exact raw actor KL, shared accepted contributions and actor-specific guards |
| `harl/algorithms/actors/happo.py` | Joint configuration validation and candidate acceptance under both caps |
| `harl/runners/on_policy_ha_runner.py` | One shared budget per round, guard dispatch and final accounting |
| `harl/runners/cascade_reservoir_runner.py` | Joint configuration and accepted/rejected budget logging |
| `examples/train_cascade.py` | `--joint-target-kl`, `--no-joint-kl-limit`, startup validation and metadata |
| `examples/compare_cascade_policy_budget.py` | Explicitly disables joint limits to preserve the original step-2/3 comparison |
| `harl/configs/algos_cfgs/happo_cascade.yaml` | Opt-in `joint_target_kl: null` |
| `tests/test_cascade_joint_kl.py` | Sixteen mathematical, transaction, sequencing, CLI and compatibility tests |
| `README.md` | Definitions, usage, logging and limitations |

Per-actor optimization records distinguish `joint_actor_kl`, `joint_kl_before`,
`joint_actor_budget`, `joint_policy_kl`, `joint_kl_remaining`,
`rejected_joint_policy_kl` and `joint_kl_early_stopped`. Local KL retains its
existing masked definition. A local-only rejection can report an in-budget
joint candidate; `joint_kl_early_stopped` distinguishes that case.

Per-actor cumulative values describe the instant that actor finishes. Read them
in `joint_kl.actor_order`, rather than reservoir index order. The round-final
`joint_kl` object includes final KL, target, remaining allowance, common row count
and actor contributions. It is included in update records and optional training
diagnostics. Joint mode emits update records even for one trajectory per update.

## Verification

**289 tests passed:** 268 core tests with the training interpreter and 21 plotting
tests with the existing plotting interpreter. The 16 new tests cover explicit
product-policy KL equality, differing actor masks, fixed references, component
replacement, prior accepted actors consuming later allowance, rollback with
fresh/populated Adam, local-only rejection, exception recovery, nonbinding exact
state/RNG equivalence, forced and zero-objective cases, guard ownership, rollout
alignment, CLI validation and retained HAPPO factors with critic training.
The complete final core suite passed after initializing the runner's diagnostic
field before its first update. `git diff --check` passed.

The cap bounds empirical mean raw command-policy movement on recorded states.
It does not bound the maximum state KL or behavior at unseen states, and is not
a monotonic-return theorem. The annual result above demonstrates that distinction.
Choosing useful limits requires a matched multi-seed comparison against the same
local rollback configuration with the joint cap disabled, with repeated sampled
evaluation and separate deterministic evaluation. The current stress test does
not supply that performance evidence.

Artifacts:

- [Raw annual records](step4_joint_kl_stress_2013_seed1_20261026.jsonl)
- [Machine-readable verification and source/data hashes](step4_joint_kl_stress_summary_20261026.json)
- [Console diagnostics](step4_joint_kl_stress_2013_seed1_20261026.console.log)

The source hashes in the summary identify the delivered source at report time;
the raw-log hash identifies the annual run's original records. Earlier step
artifacts and the pre-existing long training log were preserved.
