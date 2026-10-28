"""Training-only learning-signal measurements on complete multi-year episodes."""
import numpy as np


def signal_statistics(values):
    """Describe spread, without treating correlated daily rows as replicates."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise FloatingPointError('Non-finite learning-signal statistics')
    if not len(values):
        return dict(count=0, **dict.fromkeys(('mean', 'std', 'p05', 'median', 'p95', 'min', 'max')))
    return dict(count=int(values.size), mean=float(values.mean()), std=float(values.std(ddof=0)),
        p05=float(np.percentile(values, 5)), median=float(np.median(values)),
        p95=float(np.percentile(values, 95)), min=float(values.min()), max=float(values.max()))


def _fit(target, prediction):
    target, prediction = np.asarray(target, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    variance = float(target.var())
    return dict(rmse=float(np.sqrt(np.mean((prediction-target)**2))),
        bias=float(np.mean(prediction-target)), explained_variance=(None if variance < 1e-10 else
            float(1.-np.var(target-prediction)/variance)))


def _advantage_comparison(gae, mc):
    nonzero = (gae != 0.) & (mc != 0.)
    return dict(gae=signal_statistics(gae), monte_carlo=signal_statistics(mc),
        nonzero_comparison_count=int(nonzero.sum()),
        sign_disagreement_fraction=(float(np.mean(np.signbit(gae[nonzero]) != np.signbit(mc[nonzero])))
                                    if nonzero.any() else None))


def learning_signal_report(runner, before, post_prediction):
    """Measure frozen targets/advantages and in-sample critic fit in reward units.

    Monte Carlo here means realized reward-to-go for the collected actions,
    not the true expected value or advantage. No fitting, sampling, normalization
    state update, or policy acceptance decision occurs in this function.
    """
    cb = runner.critic_buffer
    shape = cb.rewards.shape
    if len(shape) != 3 or shape[0] < 1 or shape[1] < 1 or shape[2] != 1:
        raise ValueError('Learning-signal diagnostics require complete cascade trajectories')
    signals = (before['mc'], before['prediction'], before['gae_targets'], post_prediction)
    for value in signals:
        if np.shape(value) != shape:
            raise ValueError('Learning-signal targets and predictions must match the rollout shape')
        if not np.isfinite(value).all():
            raise FloatingPointError('Non-finite learning-signal targets or predictions')
    if (cb.masks.shape != (shape[0]+1, shape[1], 1) or
            not np.isin(cb.masks, (0., 1.)).all() or not np.all(cb.masks[-1] == 0.) or
            not np.all(cb.bad_masks == 1.)):
        raise ValueError('Learning-signal diagnostics require true terminal, untruncated episodes')
    if not np.all(cb.masks[1:-1] == 1.):
        raise ValueError('Parallel diagnostics require one complete episode per lane without autoresets')
    starts = np.zeros(shape[1], dtype=int)
    recorded_starts = getattr(runner, 'batch_episode_starts', starts)
    if not np.array_equal(starts, recorded_starts):
        raise ValueError('Learning-signal episode boundaries disagree with the rollout metadata')
    windows = runner.batch_window_starts
    source_years = runner.batch_source_years
    if len(windows) != len(starts) or len(source_years) != len(starts):
        raise ValueError('Learning-signal diagnostics require one source window per trajectory')
    ends = np.full(shape[1], shape[0], dtype=int)
    lengths = (ends-starts).tolist()
    recorded_lengths = getattr(runner, 'batch_episode_lengths', [])
    if recorded_lengths and list(recorded_lengths) != lengths:
        raise ValueError('Learning-signal episode lengths disagree with the rollout metadata')
    mc, old_prediction, targets, post_prediction = signals
    # Match the actual float32 training arithmetic before making float64 summaries.
    raw_advantage = targets-old_prediction
    mc_advantage = mc-old_prediction
    shared = runner.algo_args['algo'].get('normalize_shared_advantages', False)
    team_advantage = ((raw_advantage-raw_advantage.mean())/(raw_advantage.std()+1e-5)
                      if shared else None)
    actors = []
    for i, (actor, buffer) in enumerate(zip(runner.actor, runner.actor_buffer)):
        active = buffer.active_masks[:-1] > 0.
        if active.shape != shape:
            raise ValueError('Learning-signal actor masks must match the rollout shape')
        if not np.array_equal(buffer.masks, cb.masks):
            raise ValueError('Learning-signal actor and critic episode boundaries must match')
        if not np.isfinite(buffer.active_masks).all():
            raise FloatingPointError('Non-finite learning-signal actor masks')
        weights = buffer.discount_weights(cb.gamma)
        if actor.use_policy_active_masks:
            weights = weights*buffer.active_masks[:-1]
        objective = weights > 0.
        # train() exits before optimization for an entirely forced actor.
        if not active.any():
            objective = np.zeros_like(objective)
        if shared:
            training_advantage = team_advantage
        elif active.any():
            normalization_values = raw_advantage.copy()
            normalization_values[~active] = np.nan
            training_advantage = ((raw_advantage-np.nanmean(normalization_values))/
                                  (np.nanstd(normalization_values)+1e-5))
        else:
            training_advantage = raw_advantage
        comparison = _advantage_comparison(raw_advantage[objective], mc_advantage[objective])
        weight_sum = float(weights[objective].sum(dtype=np.float64))
        actors.append(dict(agent=i, controllable_rows=int(active.sum()), objective_rows=int(objective.sum()),
            normalization_scope='all_common_rows' if shared else 'actor_controllable_rows',
            raw_advantage=comparison['gae'], training_advantage=signal_statistics(training_advantage[objective]),
            gae_mc_sign_disagreement_fraction=comparison['sign_disagreement_fraction'],
            nonzero_comparison_count=comparison['nonzero_comparison_count'], objective_weight_sum=weight_sum,
            weighted_training_advantage_mean=(float(np.sum(
                weights[objective].astype(np.float64)*training_advantage[objective])/weight_sum)
                if weight_sum > 0. else None)))
    per_episode = []
    for lane, (window, start, end) in enumerate(zip(windows, starts, ends)):
        rows = (slice(start, end), slice(lane, lane+1), slice(None))
        per_episode.append(dict(window_start_year=int(window), source_years=list(source_years[lane]),
            episode_days=int(end-start), episode_reward=float(mc[start, lane, 0]),
            gae_start_target=float(targets[start, lane, 0]), pre_value_start=float(old_prediction[start, lane, 0]),
            critic_mc_pre=_fit(mc[rows], old_prediction[rows]),
            critic_mc_post=_fit(mc[rows], post_prediction[rows]),
            gae_target_vs_mc=_fit(mc[rows], targets[rows]),
            advantages=_advantage_comparison(raw_advantage[rows], mc_advantage[rows])))
    unique, counts = np.unique(np.asarray(windows, dtype=int), return_counts=True)
    frequencies = counts.astype(np.float64)/len(windows)
    use_gae = bool(getattr(cb, 'use_gae', runner.algo_args['algo'].get('use_gae', True)))
    return dict(scope='training_rollout', gamma=float(cb.gamma), gae_lambda=float(cb.gae_lambda),
        use_gae=use_gae, target_estimator='gae' if use_gae else 'discounted_return',
        interpretation='realized Monte Carlo targets and rowwise spread on the collected training trajectories; '
                       'not true advantages, independent daily samples, or held-out critic accuracy',
        batch=dict(episode_trajectories=len(windows), episode_horizon_days=int(shape[0]),
            parallel_environments=int(shape[1]), daily_rows=int(shape[0]*shape[1]), unique_window_count=len(unique),
            window_start_counts={str(window):int(count) for window, count in zip(unique, counts)},
            window_diversity_effective_count=float(1./np.square(frequencies).sum()),
            diversity_definition='inverse concentration of sampled-window frequencies; not statistical sample size',
            episode_lengths=lengths, trajectory_return_statistics=signal_statistics(
                [mc[start, lane, 0] for lane, start in enumerate(starts)])),
        gae_target_vs_mc=_fit(mc, targets),
        advantages=_advantage_comparison(raw_advantage, mc_advantage), actors=actors, per_episode=per_episode)
