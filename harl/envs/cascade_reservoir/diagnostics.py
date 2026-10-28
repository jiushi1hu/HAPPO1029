"""Read-only training diagnostics; no optimizer steps or result files."""
import json
import numpy as np
import torch
from harl.envs.cascade_reservoir.evaluation import evaluate_policy
from harl.envs.cascade_reservoir.reward import validate_discount_factor


def require_finite(name, value):
    if not np.isfinite(value).all():
        raise FloatingPointError('Non-finite diagnostic input: ' + name)


@torch.no_grad()
def parameter_snapshot(module):
    return {name: p.detach().clone() for name, p in module.named_parameters() if p.requires_grad}


@torch.no_grad()
def parameter_update(module, before):
    """Measure tensor updates and final backward gradients without modifying them."""
    unchanged, missing, zero = [], [], []
    delta_sq, grad_sq = 0., 0.
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        require_finite('parameter ' + name, p.detach().cpu().numpy())
        delta = p.detach() - before[name]
        delta_sq += float(delta.double().square().sum())
        if torch.equal(p, before[name]):
            unchanged.append(name)
        if p.grad is None:
            missing.append(name)
        else:
            require_finite('gradient ' + name, p.grad.detach().cpu().numpy())
            grad_sq += float(p.grad.double().square().sum())
            if torch.count_nonzero(p.grad).item() == 0:
                zero.append(name)
    return dict(parameter_delta_l2=delta_sq**.5, final_grad_norm_post_clip=grad_sq**.5,
                unchanged_parameter_tensors=unchanged, missing_gradient_tensors=missing,
                zero_gradient_tensors=zero)


def fit_metrics(target, prediction):
    target, prediction = np.asarray(target).ravel(), np.asarray(prediction).ravel()
    variance = float(np.var(target))
    return dict(rmse=float(np.sqrt(np.mean((prediction-target)**2))),
                bias=float(np.mean(prediction-target)),
                explained_variance=None if variance < 1e-10 else
                    float(1.-np.var(target-prediction)/variance))


def monte_carlo_returns(rewards, masks, gamma):
    """Realized discounted reward-to-go, with no critic bootstrap.

    The calendar rollout ends at a true terminal state. Masks also keep an
    internal terminal boundary from leaking rewards into the preceding episode.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    masks = np.asarray(masks, dtype=np.float64)
    if gamma == 1. and np.all(masks[1:-1] == 1.):
        return np.cumsum(rewards[::-1], axis=0)[::-1].copy()
    returns = np.empty_like(rewards)
    remaining = np.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        remaining = rewards[t] + gamma * masks[t+1] * remaining
        returns[t] = remaining
    return returns


@torch.no_grad()
def actor_distribution(actor, buffer):
    """Use the same stored action support as collection and optimization."""
    available = buffer.available_actions
    _, _, dist = actor.evaluate_actions(
        buffer.obs[:-1].reshape(-1, buffer.obs.shape[-1]),
        buffer.actions.reshape(-1, 1),
        None if available is None else available[:-1].reshape(-1, available.shape[-1]))
    return type(dist)(logits=dist.logits.clone())


@torch.no_grad()
def before_update(runner):
    cb = runner.critic_buffer
    for name in ('share_obs', 'value_preds', 'rewards', 'returns', 'masks', 'bad_masks'):
        require_finite('critic buffer ' + name, getattr(cb, name))
    for i, buffer in enumerate(runner.actor_buffer):
        for name in ('obs', 'actions', 'action_log_probs', 'active_masks', 'masks'):
            require_finite('actor {} buffer {}'.format(i, name), getattr(buffer, name))
    prediction = cb.value_preds[:-1].copy()
    if runner.value_normalizer is not None:
        prediction = runner.value_normalizer.denormalize(prediction)
    mc = monte_carlo_returns(cb.rewards, cb.masks, cb.gamma)
    return dict(mc=mc, prediction=prediction, gae_targets=cb.returns[:-1].copy(),
                actor_parameters=[parameter_snapshot(a.actor) for a in runner.actor],
                critic_parameters=parameter_snapshot(runner.critic.critic),
                distributions=[actor_distribution(a,b) for a,b in zip(runner.actor, runner.actor_buffer)])


@torch.no_grad()
def after_update(runner, before):
    cb = runner.critic_buffer
    starts = getattr(runner, 'batch_episode_starts', np.array([0],dtype=int))
    mc_start = float((before['mc'][0,:,0] if cb.rewards.shape[1] > 1 else before['mc'][starts,0,0]).mean())
    prediction_start = float((before['prediction'][0,:,0] if cb.rewards.shape[1] > 1 else
                              before['prediction'][starts,0,0]).mean())
    prediction = runner.critic.get_values(
        cb.share_obs[:-1].reshape(-1, cb.share_obs.shape[-1]))
    prediction = prediction.cpu().numpy().reshape(cb.rewards.shape)
    if runner.value_normalizer is not None:
        prediction = runner.value_normalizer.denormalize(prediction)
    require_finite('post-update values', prediction)
    result = dict(episode=runner.iteration, sampled_year=runner.sampled_years[-1],
        gamma=float(cb.gamma),
        update=getattr(runner,'update_count',0), training_steps=runner.training_steps,
        batch_episodes=len(starts), sampled_years=getattr(runner,'batch_years',[]),
        episode_reward=mc_start,
        discounted_episode_reward=mc_start,
        episode_energy_TWh=runner.last_rollout['episode_energy_TWh'],
        critic_update=parameter_update(runner.critic.critic, before['critic_parameters']),
        critic_mc_pre=fit_metrics(before['mc'], before['prediction']),
        critic_mc_post=fit_metrics(before['mc'], prediction),
        critic_gae_post=fit_metrics(cb.returns[:-1], prediction),
        actors=[])
    from harl.envs.cascade_reservoir.learning_signal import learning_signal_report
    result['learning_signal'] = learning_signal_report(runner, before, prediction)
    if runner.reward_settings['enabled']:
        result.update(predicted_start_return_reward_units=prediction_start,
            realized_sampled_return_reward_units=mc_start,
            episode_penalty=runner.last_reward_metrics['episode_penalty'],
            episode_generation_reward=runner.last_reward_metrics['episode_generation_reward'],
            episode_penalty_components={k:runner.last_reward_metrics['episode_penalty_'+k]
                for k in ('guaranteed_output','ecological_flow')})
    else:
        result.update(predicted_start_return_TWh=prediction_start*runner.envs[0].physics.reference_energy/1e6,
            realized_sampled_return_TWh=mc_start*runner.envs[0].physics.reference_energy/1e6)
    for i,(actor,buffer,old) in enumerate(zip(runner.actor, runner.actor_buffer,before['distributions'])):
        new = actor_distribution(actor,buffer)
        update = parameter_update(actor.actor, before['actor_parameters'][i])
        active = torch.from_numpy(buffer.active_masks[:-1].reshape(-1)>0)
        if not active.any():
            result['actors'].append(dict(agent=i, active_samples=0, **update)); continue
        action = torch.from_numpy(buffer.actions.reshape(-1,1))
        log_ratio = (new.log_probs(action)-old.log_probs(action))[active]
        ratio = log_ratio.exp()
        kl = torch.distributions.kl_divergence(old,new)[active]
        grid = torch.linspace(0., 1., new.probs.shape[-1])
        mean = (new.probs[active]*grid).sum(-1, keepdim=True)
        second = (new.probs[active]*grid.square()).sum(-1, keepdim=True)
        variance = (second-mean.square()).clamp_min(0.)
        deterministic = new.mode()[active].float()/(new.probs.shape[-1]-1)
        top_two = new.probs[active].topk(2, dim=-1).values
        require_finite('actor {} ratios'.format(i), ratio.cpu().numpy())
        require_finite('actor {} KL'.format(i), kl.cpu().numpy())
        sampled = action[active]/(new.probs.shape[-1]-1)
        result['actors'].append(dict(agent=i, active_samples=int(active.sum()),
            **update,
            entropy=float(new.entropy()[active].mean()),
            normalized_entropy=float(new.entropy()[active].mean()/np.log(new.probs.shape[-1])),
            mean_max_action_probability=float(top_two[:,0].mean()),
            mean_top_two_probability_gap=float((top_two[:,0]-top_two[:,1]).mean()),
            mode_change_fraction=float((old.mode()[active] != new.mode()[active]).float().mean()),
            sampled_action_mean=float(sampled.mean()),
            sampled_action_std=float(sampled.std(unbiased=False)),
            sampled_boundary_fraction=float(((sampled<=.01)|(sampled>=.99)).float().mean()),
            policy_kl=float(kl.mean().clamp(min=0)),
            ratio_min=float(ratio.min()),ratio_max=float(ratio.max()),
            clip_fraction=float(((ratio<1.-actor.clip_param)|(ratio>1.+actor.clip_param)).float().mean()),
            mean_u=float(mean.mean()),std_u=float(variance.sqrt().mean()),
            eval_mapping_fraction=float((deterministic**2).mean()),
            expected_sampled_mapping_fraction=float((mean**2+variance).mean())))
    return result


def print_update_window(runner, window=10):
    rows=runner.diagnostics_history[-window:]
    def avg(key,metric):
        values=[r[key][metric] for r in rows if r[key][metric] is not None]
        return float(np.mean(values)) if values else None
    active=[a for r in rows for a in r['actors'] if a['active_samples']]
    summary=dict(episode=runner.iteration,update=runner.update_count,training_steps=runner.training_steps,
        window_updates=len(rows),window_episodes=sum(r.get('batch_episodes',1) for r in rows),gamma=rows[-1]['gamma'],
        sampled_years=[year for r in rows for year in (r.get('sampled_years') or [r['sampled_year']])],
        mc_rmse_pre=avg('critic_mc_pre','rmse'),mc_rmse_post=avg('critic_mc_post','rmse'),
        mc_explained_variance_post=avg('critic_mc_post','explained_variance'),
        gae_explained_variance_post=avg('critic_gae_post','explained_variance'),
        max_actor_policy_kl=max((a['policy_kl'] for a in active),default=0.),
        mean_actor_clip_fraction=float(np.mean([a['clip_fraction'] for a in active])) if active else 0.,
        ratio_min=min((a['ratio_min'] for a in active),default=1.),
        ratio_max=max((a['ratio_max'] for a in active),default=1.),
        last_actors=rows[-1]['actors'])
    suffix = 'reward_units' if runner.reward_settings['enabled'] else 'TWh'
    for prefix in ('predicted_start_return_', 'realized_sampled_return_'):
        key = prefix+suffix
        summary[key] = float(np.mean([r[key] for r in rows]))
    if runner.reward_settings['enabled']:
        summary['last_episode_penalty'] = rows[-1]['episode_penalty']
        summary['last_episode_generation_reward'] = rows[-1]['episode_generation_reward']
        summary['last_episode_penalty_components'] = rows[-1]['episode_penalty_components']
    # Existing grad norms are measured BEFORE clipping, not applied update sizes.
    training=runner.training_history[-window:]
    summary['gradient_clip_threshold']=float(runner.algo_args['algo']['max_grad_norm'])
    summary['critic_grad_norm_pre_clip_mean']=float(np.mean([r['critic_critic_grad_norm'] for r in training]))
    summary['last_episode_reward']=rows[-1]['episode_reward']
    summary['last_discounted_episode_reward']=rows[-1]['discounted_episode_reward']
    summary['last_episode_energy_TWh']=rows[-1]['episode_energy_TWh']
    summary['last_critic_update']=rows[-1]['critic_update']
    if 'learning_signal' in rows[-1]:
        summary['last_learning_signal'] = rows[-1]['learning_signal']
    summary['last_training_metrics']={k:v for k,v in training[-1].items()
                                     if k.startswith('actor') or k.startswith('critic_')}
    summary['happo_factor_min']=min(r['actor{}_factor_min'.format(i)] for r in training for i in range(5))
    summary['happo_factor_max']=max(r['actor{}_factor_max'.format(i)] for r in training for i in range(5))
    print('DIAG_UPDATE '+json.dumps(summary),flush=True)


def _repeat_statistics(values):
    """Action-sampling statistics, conditional on one fixed evaluation-year set."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Policy diagnostics require finite nonempty repeat values')
    sd = float(values.std(ddof=1)) if len(values) > 1 else None
    return dict(mean=float(values.mean()), sd=sd,
                se=None if sd is None else sd/np.sqrt(len(values)))


def _policy_evaluation_report(runner, deterministic_rows, previous_report, scope,
                              evaluation_gamma=None):
    """Evaluate a common action-sampling grid without changing training state.

    Repeat seeds intentionally omit the checkpoint number. Differences between
    matched checkpoints therefore use common random numbers, rather than two
    unrelated evaluation samples. Their uncertainty does not measure variation
    across trained seeds or generalization to unobserved hydrologic years.
    """
    if scope not in ('validation', 'fixed_training_year', 'training_years'):
        raise ValueError('Policy diagnostic scope must be validation, fixed_training_year or training_years')
    if not deterministic_rows:
        raise ValueError('Policy diagnostics require deterministic evaluation rows')
    if all('year' in row for row in deterministic_rows):
        years = [int(row['year']) for row in deterministic_rows]
    else:
        if scope == 'training_years':
            raise ValueError('Training-year diagnostic rows must identify their evaluation years')
        # Compatibility with older direct diagnostic callers/test fixtures.
        years = (list(runner.data.validation_years) if scope == 'validation' else
                 [runner.fixed_training_year])
        if len(years) != len(deterministic_rows) or any(year is None for year in years):
            raise ValueError('Policy diagnostic rows must identify their evaluation years')
    if len(set(years)) != len(years):
        raise ValueError('Policy diagnostic years must be unique')
    if scope == 'training_years' and not set(years).issubset(runner.data.training_years):
        raise ValueError('Training-year diagnostics cannot evaluate validation or test years')
    repeats = runner.algo_args['experiment'].get('diagnostic_stochastic_repeats', 3)
    if isinstance(repeats, bool) or not isinstance(repeats, (int, np.integer)) or repeats < 1:
        raise ValueError('Policy diagnostics require a positive integer repeat count')
    repeats = int(repeats)
    training_gamma = float(runner.algo_args['algo']['gamma'])
    gamma = training_gamma if evaluation_gamma is None else validate_discount_factor(evaluation_gamma)
    if evaluation_gamma is not None and any(row.get('gamma') != gamma for row in deterministic_rows):
        raise ValueError('Deterministic rows must use the explicitly requested evaluation gamma')
    reward_samples, energy_samples, seeds = [], [], []
    by_year = [dict(year=year, repeats=[]) for year in years]
    modes = [(actor.actor, actor.actor.training) for actor in runner.actor]
    try:
        for repeat in range(repeats):
            reward_values, energy_values, repeat_seeds = [], [], []
            for j, year in enumerate(years):
                evaluation_seed = 1000000+runner.seed*100000+repeat*10000+year
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(evaluation_seed)
                    row = evaluate_policy(runner.actor, runner.data, runner.physical_config,
                        year, runner.seed, runner.iteration, runner.training_steps,
                        deterministic=False, gamma=gamma)
                reward, energy = float(row['episode_reward']), float(row['energy_TWh'])
                require_finite('sampled policy reward/energy', np.array([reward, energy]))
                penalty = row.get('episode_penalty')
                if penalty is not None:
                    penalty = float(penalty)
                    require_finite('sampled policy penalty', np.asarray(penalty))
                by_year[j]['repeats'].append(dict(repeat=repeat,
                    evaluation_seed=int(evaluation_seed), episode_reward=reward,
                    energy_TWh=energy, episode_penalty=penalty))
                if evaluation_gamma is not None:
                    by_year[j]['repeats'][-1].update(
                        episode_generation_reward=row['episode_generation_reward'],
                        service_shortfalls=row['service_shortfalls'])
                reward_values.append(reward); energy_values.append(energy)
                repeat_seeds.append(int(evaluation_seed))
            reward_samples.append(float(np.mean(reward_values)))
            energy_samples.append(float(np.mean(energy_values)))
            seeds.append(repeat_seeds)
    finally:
        for module, training in modes:
            module.train(training)
    for row, deterministic in zip(by_year, deterministic_rows):
        row['reward_statistics'] = _repeat_statistics([r['episode_reward'] for r in row['repeats']])
        row['energy_TWh_statistics'] = _repeat_statistics([r['energy_TWh'] for r in row['repeats']])
        row['deterministic_reward'] = float(deterministic['episode_reward'])
        row['deterministic_energy_TWh'] = float(deterministic['energy_TWh'])
        row['deterministic_minus_sampled_reward'] = (
            row['deterministic_reward']-row['reward_statistics']['mean'])
    reward_stats, energy_stats = _repeat_statistics(reward_samples), _repeat_statistics(energy_samples)
    deterministic_reward = float(np.mean([r['episode_reward'] for r in deterministic_rows]))
    deterministic_energy = float(np.mean([r['energy_TWh'] for r in deterministic_rows]))
    require_finite('deterministic policy reward/energy', np.array([deterministic_reward, deterministic_energy]))
    report = dict(record_type='policy_evaluation_diagnostics', seed=int(runner.seed),
        episode=runner.iteration, update=getattr(runner, 'update_count', 0),
        training_steps=runner.training_steps, gamma=gamma, scope=scope,
        evaluation_years=years, sampled_repeats=repeats, evaluation_seeds=seeds,
        evaluation_seed_rule='1000000 + training_seed*100000 + repeat*10000 + year; repeat starts at zero',
        uncertainty_scope='action sampling conditional on these evaluation years and this trained seed; '
                          'not uncertainty across training seeds or unseen years',
        evaluation_scope=('in-sample fixed training year; no generalization claim' if scope == 'fixed_training_year'
                          else 'in-sample training years; no generalization claim' if scope == 'training_years'
                          else 'held-out validation years'),
        deterministic_mean_TWh=deterministic_energy, stochastic_mean_TWh=energy_stats['mean'],
        stochastic_repeat_means_TWh=energy_samples,
        stochastic_repeat_sd_TWh=energy_stats['sd'] if energy_stats['sd'] is not None else 0.,
        stochastic_repeat_se_TWh=energy_stats['se'], years=deterministic_rows,
        deterministic_mean_reward=deterministic_reward, stochastic_mean_reward=reward_stats['mean'],
        stochastic_repeat_means_reward=reward_samples,
        stochastic_repeat_sd_reward=reward_stats['sd'] if reward_stats['sd'] is not None else 0.,
        stochastic_repeat_se_reward=reward_stats['se'], sampled_results_by_year=by_year,
        deterministic_minus_sampled_reward=deterministic_reward-reward_stats['mean'],
        deterministic_minus_sampled_TWh=deterministic_energy-energy_stats['mean'])
    if evaluation_gamma is not None:
        report['training_gamma'] = training_gamma
    comparison = dict(available=False, reason='no_previous_report')
    if previous_report is not None:
        matched = all(previous_report.get(key) == report[key] for key in (
            'seed', 'gamma', 'scope', 'evaluation_years', 'sampled_repeats', 'evaluation_seeds'))
        if not matched:
            comparison = dict(available=False, reason='previous_evaluation_grid_or_scope_does_not_match')
        else:
            reward_delta = np.asarray(reward_samples)-previous_report['stochastic_repeat_means_reward']
            energy_delta = np.asarray(energy_samples)-previous_report['stochastic_repeat_means_TWh']
            reward_delta_stats = _repeat_statistics(reward_delta)
            energy_delta_stats = _repeat_statistics(energy_delta)
            comparison = dict(available=True, previous_episode=previous_report['episode'],
                previous_update=previous_report.get('update'),
                deterministic_reward_delta=deterministic_reward-previous_report['deterministic_mean_reward'],
                deterministic_energy_TWh_delta=deterministic_energy-previous_report['deterministic_mean_TWh'],
                stochastic_repeat_reward_deltas=reward_delta.tolist(),
                stochastic_reward_delta_mean=reward_delta_stats['mean'],
                stochastic_reward_delta_sd=reward_delta_stats['sd'],
                stochastic_reward_delta_se=reward_delta_stats['se'],
                stochastic_repeat_energy_TWh_deltas=energy_delta.tolist(),
                stochastic_energy_TWh_delta_mean=energy_delta_stats['mean'],
                stochastic_energy_TWh_delta_sd=energy_delta_stats['sd'],
                stochastic_energy_TWh_delta_se=energy_delta_stats['se'])
    report['comparison_to_previous'] = comparison
    return report


def policy_evaluation_report(runner, deterministic_rows, previous_report=None, scope='validation',
                             *, evaluation_gamma=None):
    """Print sampled/modal policy evidence on an unchanged evaluation grid."""
    report = _policy_evaluation_report(runner, deterministic_rows, previous_report, scope,
                                       evaluation_gamma=evaluation_gamma)
    print('DIAG_POLICY '+json.dumps(report, allow_nan=False), flush=True)
    return report


def validation_report(runner, deterministic_rows, previous_report=None):
    """Keep the legacy validation report while adding paired policy diagnostics."""
    report = _policy_evaluation_report(runner, deterministic_rows, previous_report, 'validation')
    print('DIAG_VALIDATION '+json.dumps(report, allow_nan=False), flush=True)
    return report
