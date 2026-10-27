# Correctness corrections: pass 1

Completed the first correction pass after the source audit. The changes address
return units, misleading diagnostics, silent row loss, LP query state and
selected-checkpoint attribution. Default learning settings and the default
terminal annual GAE/unclipped-ValueNorm calculations are preserved.

## Corrected behaviors

| Issue | Before | After |
|---|---|---|
| Non-GAE endpoint with ValueNorm | Normalized next value was inserted directly into reward-unit returns | Endpoint is denormalized in both time-limit branches |
| Forced-action ratio reporting | Placeholder rows lowered the reported ratio even for an unchanged policy | Only retained positive-objective rows contribute; `ratio_samples` makes the count explicit |
| Availability-mask diagnostics | Distribution diagnostics omitted stored masks | Probabilities, entropy, modes and KL use the original action support |
| Nondividing minibatches | Low-level generators silently omitted remainder rows | Invalid counts fail before shuffle; calendar runner retains valid divisor adjustment |
| Reused LP interval queries | Earlier queries mutated the bounds used by later queries | Each query passes fresh prefix-conditioned bounds; feasibility checks retain the base problem |
| Selected-checkpoint attribution | Best episode was combined with the final run's training days | Selected episode, days and update are captured together; full run cost is separate |

The non-GAE unit reproduction now returns `226` for reward `1`, gamma `0.9`,
ValueNorm mean `200`, standard deviation `100`, and normalized next value `0.5`.
The previous result was `1.45`. This path is conditional; default GAE equations
are unchanged.

The unchanged uniform-three-action policy with two forced rows out of four now
reports `ratio=1` and `ratio_samples=2`, instead of approximately `0.666667`.
This ratio describes sampled-action ratios computed during accepted optimizer
minibatches. Contributions are counted across epochs and averaged by row count,
without treating them as independent statistical samples. Rejected candidates,
forced rows and zero-discount objective rows are excluded. If no retained
positive-objective rows exist, ratio and count remain zero by convention; the
ratio is undefined in that case. Loss averaging and optimizer decisions are
preserved.

A categorical diagnostic with one of three actions unavailable now reports
probabilities `[0, 0.5, 0.5]` and entropy approximately `0.693147`, matching the
training policy. Previously it reported the unrestricted uniform distribution
and entropy approximately `1.098612`. Unrestricted buffers may also use
`available_actions=None` through actor minibatch generation.

For the two-day LP reproduction, querying reservoir 1 and then reservoir 0 no
longer changes reservoir 0's interval from `[0, 2]` to `[0, 0]`. Repeated,
out-of-order and failed queries leave the base bounds intact. Existing monotone
upstream-to-downstream queries match the independently constructed original LP
exactly. The fast path, terminal tolerance recovery and physical constraints are
preserved.

The checkpoint-attribution reproduction now reports seed 1's selected episode
294 with `training_steps=107374` and `selected_update=49`. The final run cost
`109567` is preserved separately as `run_training_steps`. An untrained selected
checkpoint retains zero steps and update zero. Both fixed-year and validation
selection capture their metadata together. Run results expose
`best_training_steps` and `best_update`.

## Clipped value loss and ValueNorm

The combination `use_clipped_value_loss=True` and an active ValueNorm is now
explicitly unsupported and rejected before mutation. The CLI rejects the
configuration before data loading; direct runner construction rejects it before
model/RNG work. Critic `cal_value_loss()`, `update()` and `train()` independently
guard it before their forwards, normalization, gradient or optimizer work.

Rebasing old predictions after updating normalization alone would not be a
complete correction: current model outputs would acquire a different physical
interpretation before a parameter update, potentially saturating the clipped
loss. Full support requires an output-preserving normalization design and its
optimizer semantics. This pass does not introduce that algorithmic change.

The supported combinations remain:

| Clipped value loss | ValueNorm | Status |
|---|---|---|
| Off | On | Existing default; calculation and update sequence preserved |
| Off | Off | Supported |
| On | Off | Supported; checked against independent loss/gradient calculations |
| On | On | Clear early error |

## Files and tests

| Source | Correction |
|---|---|
| `harl/common/buffers/on_policy_critic_buffer_ep.py` | Reward-unit non-GAE bootstrap and critic minibatch validation |
| `harl/algorithms/critics/v_critic.py` | Explicit normalization/clipping compatibility guard |
| `harl/common/buffers/on_policy_actor_buffer.py` | Actor minibatch validation and optional availability masks |
| `harl/algorithms/actors/happo.py` | Effective-row ratio reporting and contribution count |
| `harl/envs/cascade_reservoir/diagnostics.py` | Availability-aware distribution diagnostics |
| `harl/envs/cascade_reservoir/terminal_storage.py` | Independent LP query bounds |
| `harl/runners/cascade_reservoir_runner.py` | Early validation and selected-checkpoint metadata |
| `examples/train_cascade.py` | Pre-data rejection of incompatible critic settings |

**343 tests passed:** 322 core tests with the training interpreter and 21 plotting
tests with the plotting interpreter. The 33 new regression tests are:

- `tests/test_critic_normalization.py`: 10 tests.
- `tests/test_policy_diagnostic_correctness.py`: 8 tests.
- `tests/test_terminal_query_state.py`: 7 tests.
- `tests/test_correction_integration.py`: 8 tests.

The default unclipped critic is checked against an independent replay of the
previous calculation over three epochs and two minibatches: parameters, Adam,
gradients, six ValueNorm updates, metrics and RNG match exactly. Policy diagnostic
measurement changes are also tested for parameter/Adam/gradient/buffer/RNG
invariance. Valid minibatches preserve the shuffled row order; invalid requests
leave shuffle RNG unchanged. `git diff --check` passed.

## Scope and next correction pass

This pass does not claim improved training performance. The existing result logs
remain historical baseline records and are not rewritten. Previously reported
active-only endpoint ratios remain valid; the corrected optimizer-path ratio has
a more precise definition.

Durable model checkpoints, resumption, complete configuration/provenance records,
completion markers and persisted per-year final-test results remain for the next
pass. The existing future-inflow feasibility oracle, physical assumptions,
reward definition, action bins, gamma, lambda, entropy coefficient, learning
rates, epoch budgets and default KL switches are unchanged.

Already-running processes retain their loaded implementation; these corrections
apply to newly started processes.
