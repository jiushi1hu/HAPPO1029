# Policy optimization budget smoke check — 2026-10-25

This is a single-seed, single-training-year, single-update diagnostic. Both arms used 2013, seed 1, one complete 365-day training episode, one sampled evaluation repeat and the same evaluation RNG seed. This check used **one annual trajectory per update**, rather than the six-year batch of the main experiment.

The five actors, 50-bin action grid, learning rates, critic budget, physical constraints, source service thresholds, unit penalty weights, gamma 0.99 and GAE lambda 0.95 were unchanged.

| Arm | Initial sampled return | Final sampled return | Initial deterministic return | Final deterministic return | Actor optimizer steps |
|---|---:|---:|---:|---:|---|
| baseline | -40.819962 | -38.922783 | -24.677927 | -35.338903 | 5, 5, 5, 5, 5 |
| guarded | -40.819962 | -35.151525 | -24.677927 | -31.649479 | 8, 10, 10, 8, 8 |

Baseline: five actor epochs without KL stopping. Guarded: at most twenty epochs, stopping further epochs when mean exact categorical KL(old || new) over controllable rollout states exceeds 0.01. All five guarded actors triggered early stopping. The last optimizer step may cross the threshold.

The guarded arm improved the sampled return more in this particular check. **Deterministic return deteriorated in both arms.** This is not evidence of stable convergence, held-out improvement or a statistically established advantage. The default remains five epochs with KL stopping disabled; the guarded profile is opt-in.

Reproduce with a new output path:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python examples/compare_cascade_policy_budget.py \
  --year 2013 --updates 1 --rollout-episodes 1 --seeds 1 \
  --evaluation-repeats 1 --output runs/policy_budget_smoke_reproduction.jsonl
```

Raw comparison records: `runs/policy_budget_smoke_20261025.jsonl`.
