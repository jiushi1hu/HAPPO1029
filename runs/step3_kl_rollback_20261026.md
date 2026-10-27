# Step 3: reject and restore KL-crossing actor updates

Implemented an opt-in candidate-step acceptance check. Enable it with
`--target-kl LIMIT --kl-rollback`, or `algo.kl_rollback: true` with a positive
finite `algo.target_kl` in YAML. The default remains `kl_rollback: false` and
`target_kl: null`. Existing KL early stopping still retains the crossing step
unless rollback is explicitly enabled.

## Accepted-state semantics

Each actor keeps one categorical reference distribution from before its entire
training update. Before an effective candidate optimizer step, the implementation
deep-copies actor state, complete Adam state, and gradients. After the step it
checks finite metrics/gradients/parameters/Adam tensors and measures exact
KL(old || candidate) over all controllable original-rollout states.

If KL exceeds the budget, the actor, Adam moments/counters, and gradients are
restored to their immediate pre-candidate values. The remaining minibatches and
epochs for that actor are stopped. Previously accepted steps remain. There is no
automatic retry, backtracking schedule, or learning-rate change.

Exceptions or nonfinite candidate state restore the transaction before the error
is propagated. Rejected candidate losses are excluded from retained-update
statistics. Forced and zero-objective minibatches do not attempt optimizer steps
or take snapshots. Legacy epoch-shuffle RNG consumption is preserved for an
entirely zero-objective rollout. A nonbinding rollback setting therefore preserves
learning state and RNG behavior; actual early termination can skip later shuffle
calls and change subsequent action-sampling streams.

The check occurs after each optimizer step, including multiple minibatches per
epoch. `ppo_epochs_completed` counts only fully processed epochs. A partially
accepted epoch can have retained optimizer steps while its completed-epoch count
is zero. Optimizer attempts, retained steps and rejected steps are logged separately.

The HAPPO runner recomputes likelihoods from the retained weights after actor
training. Sequential factors therefore incorporate accepted policy changes only.
An actor rejecting its first candidate contributes likelihood ratio one. Critic
updates still occur after actor optimization as before.

## Real annual correctness stress check

The actual 2013 reservoir environment was run for one complete annual training
episode with seed 1, 50 action bins, default 128/128 networks and learning rates,
five maximum actor epochs, seasonal constraints, variable head, service penalties,
and annual storage closure. Rollback used a deliberately tiny `target_kl=1e-8`
to exercise rejection. This is a correctness test, not a calibrated threshold or
an experiment claiming better returns.

| Actor | Rejected candidate KL | Retained KL | Attempts | Retained steps | Rejected steps |
|---|---:|---:|---:|---:|---:|
| WDD | 0.000178800 | 0 | 1 | 0 | 1 |
| BHT | 0.000156488 | 0 | 1 | 0 | 1 |
| XLD | 0.000140919 | 0 | 1 | 0 | 1 |
| XJB | 0.000188320 | 0 | 1 | 0 | 1 |
| THR | 0.000186700 | 0 | 1 | 0 | 1 |

All five first candidates were rejected. Every actor had zero parameter change,
zero clipped-surrogate change, and zero mode-change fraction. HAPPO's factor
minimum and maximum both remained exactly one. The critic still updated, with
parameter-change L2 norm 0.206294.

Deterministic net return remained exactly -24.677926783757048 before and after.
The one matched sampled evaluation remained exactly -37.2532844058023. One
sampled repeat has no estimated sampling standard error; it is used here only
as an equality check on the unchanged policy. The unit tests separately verify
exact restoration with both fresh and already-populated Adam momentum.

Artifacts:

- [Raw annual records](step3_kl_rollback_stress_2013_seed1_20261026.jsonl)
- [Machine-readable verification summary](step3_kl_rollback_stress_summary_20261026.json)
- [Console diagnostics](step3_kl_rollback_stress_2013_seed1_20261026.console.log)

## Verification and scope

273 regression tests passed on the delivered behavior: 252 core tests with the
training interpreter and 21 plotting tests with the existing plotting interpreter.
The 15 new rollback tests cover exact restoration of parameters, Adam moments and
counters, and gradients; partial acceptance; the fixed KL reference; per-minibatch
stopping; zero-objective RNG equivalence; exceptions/nonfinite candidates; legacy
behavior; accepted-weight sequential factors; CLI validation; and comparison/log
integration. The final full core run followed the zero-objective RNG correction.
`git diff --check` passed.

The main changes are in `harl/algorithms/actors/happo.py`, configuration and CLI
validation, runner reporting, and `tests/test_cascade_kl_rollback.py`. The comparison
driver accepts `--kl-only --candidate-kl-rollback` for a five-versus-five rollback
comparison, with metadata identifying both the KL budget and rollback switch.
Plain `--kl-only` retains the step-2 after-epoch comparison. Earlier result files
were preserved.

This safeguards the measured mean KL on collected controllable states. It does
not impose a maximum over every possible state, a joint five-actor KL budget,
or monotonic true-return improvement. The step-2 result remains relevant: a
0.01 budget can be nonbinding at five epochs. No claim of improved convergence
or performance follows from this forced-rejection check. Reward, physics,
clipping, entropy, learning rates, and normal training defaults were not changed.

## Usage

To enable rollback with the previously discussed budget, add:

```bash
--target-kl 0.01 --kl-rollback
```

To reproduce the correctness stress check, choose a new log filename:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python -B examples/train_cascade.py \
  --fixed-training-year 2013 --episodes 1 --rollout-episodes 1 --eval-interval 1 \
  --seeds 1 --action-bins 50 --ppo-epochs 5 --target-kl 1e-8 --kl-rollback \
  --diagnostics --diagnostic-episodes 0 1 --diagnostic-stochastic-repeats 1 --no-plot \
  --episode-log runs/step3_kl_rollback_stress_reproduction.jsonl
```
