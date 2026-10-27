# Assessment of the supplied training results

The supplied results show substantial sampled-policy learning, improving critic
fit, and reduced service penalties. They also show a persistent gap between
sampled and deterministic behavior, nonmonotonic deterministic validation, and
substantial remaining penalties. These records do not establish convergence,
optimality, or the benefit of the optional KL safeguards.

This assessment uses the fixed attachment supplied on October 27, 2026:
1,172 valid JSON records, 11,215,702 bytes, SHA-256
`3a4877d6a6c0e3d7b07fabb4ecf5eeaa74f7c0c9b5847eadd98157626350332f`.
The repository log had identical bytes when inspected. No training code or
original result log was modified.

## Snapshot coverage and actual configuration

| Seed | Annual training episodes | Updates | Latest deterministic validation | Latest sampled validation |
|---|---:|---:|---:|---:|
| 1 | 300 | 50 | Episode 300 | Episode 300 |
| 2 | 300 | 50 | Episode 300 | Episode 300 |
| 3 | 168 | 28 | Episode 162 | Episode 120 |

Seeds 4 and 5 are absent from this snapshot. If the five-seed launch command was
used, it is not a complete five-seed result. The last record is seed 3's training
diagnostic after episode 168; it precedes a corresponding validation record.
This is not evidence of a crash. The attachment alone does not show whether the
process was still running when copied.

The recorded configuration is 200 action bins, gamma 0.99, GAE lambda 0.95,
six annual episodes per update, and five maximum actor epochs. Every actor
retained five optimizer steps in each logged update. The relevant switches are:

```
actor_target_kl = null
actor_kl_rollback = false
joint_target_kl = null
```

This run exercises the new diagnostics with the baseline optimizer. It does not
experimentally test the step-3 rollback or step-4 joint cap, and it does not test
lambda one. The earlier bounded lambda comparison used 50 bins, so it is not a
matched control for this 200-bin run.

## Evaluation demonstrates learning

Deterministic validation net return is averaged over years 2020–2024:

| Seed | Initial | Best observed | Best episode | Latest | Latest episode |
|---|---:|---:|---:|---:|---:|
| 1 | -9.1529 | +5.0427 | 294 | +4.4044 | 300 |
| 2 | -23.8743 | +0.4066 | 300 | +0.4066 | 300 |
| 3 | -22.3224 | +4.3472 | 108 | -5.6420 | 162 |

The completed seeds improve on their initialization. Seed 3 has a large decline
from its best modal checkpoint, approximately 9.99 reward units by episode 162.
Its latest sampled evaluation is only at episode 120, so the attachment cannot
establish that its sampled return deteriorated during the same interval.

Repeated sampled evaluation provides a different and more consistently improving
picture:

| Seed | Endpoint | Initial sampled mean | Endpoint sampled mean | Paired improvement | Paired sampling SE |
|---|---:|---:|---:|---:|---:|
| 1 | 300 | -49.2749 | -31.9079 | +17.3670 | 0.1799 |
| 2 | 300 | -49.5923 | -34.3429 | +15.2495 | 0.7618 |
| 3 | 120 | -49.7632 | -45.9200 | +3.8432 | 0.4961 |

These differences use three matched action-repeat means on the same five
validation years and the same per-year evaluation seeds. All three initial-to-
endpoint repeat differences are positive for each seed. Every individual
validation year's sampled mean also improves in each available comparison.
Thus the improvement is not explained merely by sampling more favorable training
years at later updates.

The SEs quantify action-sampling variability conditional on the selected
validation years and one trained seed. They are not uncertainty over trained
seeds, unseen hydrologic years or a new deployment setting. Three repeats are
still a small evaluation sample.

For the completed seeds, milestone sampled means at episodes
0/24/60/120/180/240/300 are:

- Seed 1: -49.27, -49.36, -49.15, -45.41, -42.04, -38.05, -31.91.
- Seed 2: -49.59, -48.40, -46.84, -44.79, -42.66, -39.51, -34.34.

The sampled curves are still improving at the final recorded milestone. A
plateau or convergence has not been demonstrated.

The sampled report at episode 300 concerns the current policy. For seed 1 it
does not measure the separately selected best policy from episode 294. The
runner retains validation-selected actor weights during the run and restores
them in `final_test()`; the JSONL contains metrics rather than those weights.

## The critic is learning

Compare the first five and last five updates of the completed seeds:

| Mean metric | Seed 1, first five → last five | Seed 2, first five → last five |
|---|---:|---:|
| Post-update critic/MC RMSE | 26.915 → 7.803 | 26.135 → 9.007 |
| Post-update critic/MC explained variance | 0.109 → 0.889 | 0.205 → 0.867 |
| GAE-target/MC gap RMSE | 24.093 → 6.152 | 24.212 → 7.302 |
| Raw GAE/MC sign disagreement | 24.70% → 23.16% | 23.48% → 19.59% |

Latest post-update MC explained variance is approximately 0.866, 0.856 and 0.916
for seeds 1–3. These are strong improvements over initialization. The data do
not support describing the critic as generally unable to learn.

Latest fit to GAE targets is even stronger, with explained variance near
0.978/0.977/0.973. That is not the same as matching realized full returns.
Residual target gaps and roughly 20–24% sign disagreement remain informative,
but realized MC advantages are noisy sampled comparators, not true advantages.
The disagreement rate alone does not prove that GAE directions are wrong.

All these critic-fit measurements use the collected training rows. They are
in-sample, and updated ValueNorm statistics contribute to post-update prediction
changes. The first/last windows contain different sampled trajectories, so the
table is a learning-trend description rather than a fixed-dataset experiment.
There is no current evidence requiring an automatic increase in critic epochs.

## Policy changes and the sampled–modal gap

All 640 actor/update combinations have positive parameter changes and positive
fixed-batch clipped-surrogate deltas, with no reported missing or zero-gradient
tensors. There is no optimizer-stall evidence. Every actor retained five steps;
there were no rejected steps because the guards were disabled.

The maximum reported actor mean KL is 0.0085627. Logged sampled likelihood ratios
span approximately 0.6635–1.5479, and the maximum actor clip fraction is 21.1%.
Average clip fractions over actor updates are much lower: approximately 1.64%,
1.23% and 0.64% for seeds 1–3.

These endpoints do not show exploding mean KL. They also do not prove an
intermediate-step, maximum-state or common-state joint KL bound. A local limit
of 0.01 exceeds all these reported endpoint means; intermediate candidate KLs
were not recorded, so its exact effect cannot be reconstructed. Positive
surrogate changes remain compatible with nonmonotonic evaluation returns.

Last-five-update policy summaries for the completed seeds are:

| Metric | Seed 1 | Seed 2 |
|---|---:|---:|
| Mean normalized entropy | 0.8903 | 0.8977 |
| Mean highest action probability | 0.0651 | 0.0900 |
| Mean mode-change fraction per update | 14.6% | 9.0% |

Uniform 200-bin policies have maximum probability 0.005. The learned policies
are concentrating, but remain broad. Their preferred actions can still change
at many recorded states per update.

At episode 300, deterministic-minus-sampled net return is 36.3123 for seed 1 and
34.7495 for seed 2. This is a substantive operational difference between modal
execution and sampling from the trained policy. A fluctuating modal curve alone
does not establish failure of sampled-policy learning. Conversely, improving
sampled performance does not ensure stable modal operation.

## Reward components and year variation

Most net-reward improvement comes from reducing service penalties:

| Modal validation component | Seed 1, initial → final | Seed 2, initial → final |
|---|---:|---:|
| Discounted generation reward | 32.4599 → 35.5649 | 33.8476 → 33.8234 |
| Discounted total service penalty | 41.6128 → 31.1606 | 57.7219 → 33.4168 |
| Actual annual generation, TWh | 194.342 → 216.910 | 212.556 → 219.369 |

In particular, seed 2 improves net reward by about 24.28 units while discounted
generation reward is nearly unchanged. Reward improvement therefore should not
be described solely as generation improvement. Discounted generation reward
also differs from undiscounted annual TWh.

Sampled penalties fall from 77.2556 to 61.2870 for seed 1 and from 77.5488 to
63.3898 for seed 2. They still exceed sampled discounted generation rewards
29.3791 and 29.0469, explaining the remaining negative sampled net returns.
Negative return is valid under the implemented reward definition.

The final deterministic five-year means also hide year-specific weaknesses:

| Validation year | Seed 1 final return | Seed 2 final return |
|---|---:|---:|
| 2020 | -4.4939 | -9.1012 |
| 2021 | +15.0196 | +10.9187 |
| 2022 | +9.2042 | +6.1666 |
| 2023 | +12.0394 | +8.5241 |
| 2024 | -9.7473 | -14.4752 |

A positive mean does not establish uniformly good service performance.

## Coverage, feasibility and limits

The six-trajectory batches average approximately 5.78/5.72/5.79 distinct years.
Completed seeds visited all 50 training years, and seed 3 visited 48 by episode
168. There is no obvious year-coverage failure that automatically calls for a
larger batch.

All 768 recorded annual training episodes completed without a reported hard
failure. All 650 deterministic validation year evaluations report feasibility
and zero physical violations; the largest reported water-balance residual is
approximately 0.000003815 cubic metres. Hard feasibility is enforced by the
environment and is separate from the substantial soft service deficits.

No final-test claim is supported by this attachment. The snapshot is incomplete,
and the source prints final test summaries to the console without persisting
their result rows to this JSONL. Absence of a test-record type alone is therefore
not proof that testing failed. Repeated validation and checkpoint selection also
make the selected best validation score an optimistic estimate of independent
performance. Untouched test years remain necessary for the final assessment.

## Practical next step

Keep this run as the documented 200-bin baseline and let the configured seeds
finish. Evaluate all completed seeds before making another optimizer change.
Current evidence supports learning with remaining performance and evaluation
gaps; it does not identify clipping strength or persistent critic underfitting
as the primary failure.

If a lambda-one experiment is performed next, retain 200 bins, the same year
pool/schedules, annual episode budget, seeds, networks and KL settings, and
change lambda alone. Compare repeated sampled return, deterministic return and
service deficits separately. A larger-batch comparison is another experiment,
with its changed update frequency explicitly recorded. Neither experiment is
an established fix from these results, and the earlier 50-bin smoke trial is
not evidence for changing the present default automatically.

Artifacts:

- [Comparison chart](train_step5_results_analysis_20261027.png)
- [Machine-readable metrics and provenance](train_step5_results_analysis_20261027.json)
