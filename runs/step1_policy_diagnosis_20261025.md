# Step 1: distinguish surrogate, sampled return, and deterministic return

The diagnostic pipeline is implemented and verified. Optimizer settings, actor
updates, reward, physics, year sampling, and deterministic checkpoint selection
are unchanged. New measurements are enabled only with `--diagnostics`.

## New bounded experiment

This is an in-sample measurement check, not evidence of convergence or held-out
improvement. It uses training year 2013, seed 1, six independent annual rollouts
under a frozen policy, and one sequential HAPPO update. Networks use the default
128/128 hidden layers, five actor epochs, five critic epochs, learning rates
0.0003, gamma 0.99, GAE lambda 0.95, clipping 0.2, entropy coefficient 0.001, and
no KL stopping. The 50-bin grid matches the existing service experiment; the
repository's current YAML default remains 200 bins. Seasonal constraints,
storage-dependent head, annual initial-storage closure, and unit service penalty
weights remain enabled.

Before and after the update, the same year is evaluated using categorical modes
and three sampled-policy repeats. Each repeat uses identical action-sampling
seeds across checkpoints. No validation or test years are evaluated in this run.

Raw records: [step1_policy_diagnosis_2013_seed1.jsonl](step1_policy_diagnosis_2013_seed1.jsonl).
Startup configuration and console output:
[step1_policy_diagnosis_2013_seed1.console.log](step1_policy_diagnosis_2013_seed1.console.log).

| Measurement | Before | After | Change |
|---|---:|---:|---:|
| Deterministic discounted net return | -24.677927 | -0.371664 | +24.306263 |
| Sampled mean discounted net return, three repeats | -38.176600 | -38.295759 | -0.119159 |
| Deterministic generation, TWh | 228.513009 | 237.676901 | +9.163892 |
| Sampled mean generation, TWh | 195.737373 | 196.618383 | +0.881009 |
| Deterministic minus sampled net return | 13.498673 | 37.924094 | +24.425421 |

The three paired sampled-return changes are -0.245840, +0.271420 and -0.383057.
Their sample SD is 0.345139 and standard error is 0.199266. The small negative
mean change does not establish expected-return deterioration. These statistics
describe action-sampling uncertainty conditional on this year and trained seed;
they are not confidence intervals over independent training seeds or hydrologic
generalization. Three repeats are a small diagnostic sample.

All five actors improved their own fixed-batch clipped surrogate and
entropy-regularized objective:

| Actor | Clipped surrogate before | After | Change | Mean categorical KL | Mode-change fraction |
|---|---:|---:|---:|---:|---:|
| WDD | 0.003664352 | 0.007375082 | +0.003710730 | 0.004202 | 0.663139 |
| BHT | 0.016745558 | 0.020160647 | +0.003415089 | 0.003177 | 0.977023 |
| XLD | 0.013686674 | 0.017213298 | +0.003526624 | 0.003323 | 0.997359 |
| XJB | -0.029660601 | -0.026911169 | +0.002749432 | 0.003720 | 0.980168 |
| THR | -0.029551579 | -0.026528399 | +0.003023179 | 0.004898 | 0.879917 |

Each surrogate comparison uses that actor's actual fixed sequential factor,
post-normalization advantages, rollout old likelihoods, action-time control masks,
per-episode discount weights, and full-row mean. Values from different actors
are not one common team-return estimate. Existing averaged policy-loss fields
measure the path through optimization, rather than these endpoints.

Post-update normalized actor entropy ranges from 0.99874 to 0.99917, close to
the uniform distribution. Despite modest KL, preferred actions changed on
66.3% to 99.7% of controllable old-rollout states. This is direct evidence that
categorical modes can change sharply while the full distribution changes only
slightly. In this particular update the modal return improved; the observation
does not imply that future mode changes will help.

Critic MC RMSE decreased from 43.0727 to 35.9862, but post-update MC explained
variance remains negative (-0.0785). One initial critic update has not established
well-calibrated advantages. A larger training surrogate alone cannot certify a
higher expected return.

## Existing longer-run evidence

The existing `service_batch6_300ep_seeds1_2_3_run1` logs were inspected through
the first 40 unique completed updates, annual episode 240, seed 1. Their console
contains an earlier restarted segment; raw line counts therefore overcount early
updates. These are 50-bin, six-year-batch, gamma-0.99 service-penalty results, not
completed multi-seed evidence or a 200-bin experiment. Existing files were not edited.

| Annual episode | Deterministic validation net return | Sampled validation mean | Sampled repeat SD |
|---|---:|---:|---:|
| 0 | -38.4506 | -50.8114 | 0.8730 |
| 24 | -2.3888 | -50.0662 | 0.8897 |
| 102 | 4.7700 | -42.8830 | 0.3831 |
| 150 | 9.1540 | -39.9278 | 1.0541 |

Both policies improve overall, while a large sampled/modal gap persists.
Among the first forty deterministic validation transitions, eleven decline.
The largest decline is episode 36 to 42: return 9.9949 to -2.8616, a change of
-12.8566. That update's maximum actor mean KL is only 0.001798. It therefore
does not require a large reported mean KL to produce a large modal decline.
There is no matched sampled evaluation at episode 42, so the expected-return
direction for that particular update cannot be recovered.

Across those forty updates, maximum actor mean KL is 0.001605 to 0.008226,
sampled likelihood ratios span 0.679287 to 1.384606, and sequential factors span
0.391637 to 2.428130. These do not establish runaway policy movement as the
main failure mechanism. Before/after surrogate endpoints and policy snapshots
are absent from the old run and cannot be reconstructed from averaged losses.

## Implemented measurements

- `HAPPO.train(..., diagnostics=True)` records before/after/delta values for
  clipped surrogate, unclipped surrogate, weighted entropy, and regularized
  objective. Measurements do not shuffle data or invoke optimizer operations.
- `training_diagnostics` records persist those endpoints under each actor's
  `surrogate` field, alongside existing critic and distribution diagnostics.
- `policy_evaluation_diagnostics` records retain each year/repeat return,
  generation, penalty and action seed; sampled/modal gaps; and paired changes
  relative to the previous compatible diagnostic checkpoint.
- Repeat SD/SE describe action sampling only. Single-repeat SE is null.
  Mismatched year grids, gamma, scope, repeat counts or training/action seeds
  do not receive a paired-delta claim.
- Fixed-year diagnostics evaluate that training year only. Held-out diagnostics
  retain their existing milestone schedule. Pair comparisons can span multiple
  updates; adjacent milestones are needed to isolate an individual update.
- JSONL diagnostics flush through the existing optional episode log. Neither
  their values nor extra sampled returns affect checkpoint selection.

## Verification

247 regression tests passed: 226 algorithm/environment/integration tests using
`/opt/anaconda3/envs/harl/bin/python`, and 21 plotting tests using
`/Library/Frameworks/Python.framework/Versions/3.9/bin/python3`. The training
interpreter lacks Matplotlib; the plotting interpreter is the repository's
documented separate plotting runtime. No dependencies were installed.

The new tests independently verify surrogate arithmetic, masks, annual discount
resets, normalization and sequential factors, all-forced/zero-weight batches,
paired uncertainty, mode/RNG restoration, and JSONL integration. Enabling the
measurements produced bit-identical actor/critic parameters, optimizer state,
gradients, ValueNorm, rollout data and random states in the invariance checks.
The real annual experiment completed six training episodes and one update with
finite diagnostics. Syntax checks and `git diff --check` also passed.

## Interpretation for the next decision

This step fills the measurement gap. The bounded update improved every actor's
sampled surrogate and greatly improved its modal controller, while the sampled
mean net return remained unresolved within sampling uncertainty. Together with
the older run's fluctuations at low mean KL, the evidence supports monitoring
sampled and modal behavior separately before tightening policy updates.
No KL threshold, clipping strength, learning rate, entropy setting, critic budget,
reward coefficient, or physical assumption has been changed. More adjacent
checkpoints and repeated evaluations across seeds/years are needed to attribute
systematic expected-return deterioration or establish a convergence remedy.
