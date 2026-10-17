"""Read-only training diagnostics; no optimizer steps or result files."""
import json
import numpy as np
import torch
from harl.envs.cascade_reservoir.evaluation import evaluate_policy


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
    _, _, dist = actor.evaluate_actions(
        buffer.obs[:-1].reshape(-1, buffer.obs.shape[-1]),
        buffer.actions.reshape(-1, 1))
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
    return dict(mc=mc, prediction=prediction,
                actor_parameters=[parameter_snapshot(a.actor) for a in runner.actor],
                critic_parameters=parameter_snapshot(runner.critic.critic),
                distributions=[actor_distribution(a,b) for a,b in zip(runner.actor, runner.actor_buffer)])


@torch.no_grad()
def after_update(runner, before):
    cb = runner.critic_buffer
    prediction = runner.critic.get_values(
        cb.share_obs[:-1].reshape(-1, cb.share_obs.shape[-1]))
    prediction = prediction.cpu().numpy().reshape(cb.rewards.shape)
    if runner.value_normalizer is not None:
        prediction = runner.value_normalizer.denormalize(prediction)
    require_finite('post-update values', prediction)
    result = dict(episode=runner.iteration, sampled_year=runner.sampled_years[-1],
        gamma=float(cb.gamma),
        episode_reward=float(cb.rewards.astype(np.float64).sum()),
        discounted_episode_reward=float(before['mc'][0,0,0]),
        episode_energy_TWh=runner.last_rollout['episode_energy_TWh'],
        critic_update=parameter_update(runner.critic.critic, before['critic_parameters']),
        critic_mc_pre=fit_metrics(before['mc'], before['prediction']),
        critic_mc_post=fit_metrics(before['mc'], prediction),
        critic_gae_post=fit_metrics(cb.returns[:-1], prediction),
        actors=[])
    if runner.reward_settings['enabled']:
        result.update(predicted_start_return_reward_units=float(before['prediction'][0,0,0]),
            realized_sampled_return_reward_units=float(before['mc'][0,0,0]),
            episode_penalty=runner.last_reward_metrics['episode_penalty'],
            episode_generation_reward=runner.last_reward_metrics['episode_generation_reward'],
            episode_penalty_components={k:runner.last_reward_metrics['episode_penalty_'+k]
                for k in ('target_band','daily_rise','daily_fall')})
    else:
        result.update(predicted_start_return_TWh=float(before['prediction'][0,0,0]*runner.envs[0].physics.reference_energy/1e6),
            realized_sampled_return_TWh=float(before['mc'][0,0,0]*runner.envs[0].physics.reference_energy/1e6))
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
        require_finite('actor {} ratios'.format(i), ratio.cpu().numpy())
        require_finite('actor {} KL'.format(i), kl.cpu().numpy())
        sampled = action[active]/(new.probs.shape[-1]-1)
        result['actors'].append(dict(agent=i, active_samples=int(active.sum()),
            **update,
            entropy=float(new.entropy()[active].mean()),
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
    summary=dict(episode=runner.iteration,window_episodes=len(rows),gamma=rows[-1]['gamma'],
        sampled_years=[r['sampled_year'] for r in rows],
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
    summary['last_training_metrics']={k:v for k,v in training[-1].items()
                                     if k.startswith('actor') or k.startswith('critic_')}
    summary['happo_factor_min']=min(r['actor{}_factor_min'.format(i)] for r in training for i in range(5))
    summary['happo_factor_max']=max(r['actor{}_factor_max'.format(i)] for r in training for i in range(5))
    print('DIAG_UPDATE '+json.dumps(summary),flush=True)


def validation_report(runner, deterministic_rows):
    """Compare deterministic actions with sampled policy, preserving RNG state."""
    repeats=int(runner.algo_args['experiment'].get('diagnostic_stochastic_repeats',3))
    samples=[]
    reward_samples=[]
    for repeat in range(repeats):
        values=[]; rewards=[]
        for year in runner.data.validation_years:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(1000000+runner.seed*100000+repeat*10000+year)
                row=evaluate_policy(runner.actor,runner.data,runner.physical_config,year,
                    runner.seed,runner.iteration,runner.training_steps,deterministic=False)
            values.append(row['energy_TWh'])
            if runner.reward_settings['enabled']:
                rewards.append(row['episode_reward'])
        samples.append(float(np.mean(values)))
        if rewards:reward_samples.append(float(np.mean(rewards)))
    report=dict(episode=runner.iteration,
        deterministic_mean_TWh=float(np.mean([r['energy_TWh'] for r in deterministic_rows])),
        stochastic_mean_TWh=float(np.mean(samples)),
        stochastic_repeat_means_TWh=samples,
        stochastic_repeat_sd_TWh=float(np.std(samples,ddof=1)) if repeats>1 else 0.,
        years=deterministic_rows)
    if runner.reward_settings['enabled']:
        report.update(deterministic_mean_reward=float(np.mean([r['episode_reward'] for r in deterministic_rows])),
            stochastic_mean_reward=float(np.mean(reward_samples)),
            stochastic_repeat_means_reward=reward_samples,
            stochastic_repeat_sd_reward=float(np.std(reward_samples,ddof=1)) if repeats>1 else 0.)
    print('DIAG_VALIDATION '+json.dumps(report),flush=True)
    return report
