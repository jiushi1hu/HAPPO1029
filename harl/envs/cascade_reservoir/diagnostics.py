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


@torch.no_grad()
def actor_distribution(actor, buffer):
    _, _, dist = actor.evaluate_actions(
        buffer.obs[:-1].reshape(-1, buffer.obs.shape[-1]), buffer.rnn_states[0],
        buffer.actions.reshape(-1, 1), buffer.masks[:-1].reshape(-1, 1))
    return type(dist)(dist.concentration1.clone(), dist.concentration0.clone())


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
    # Full realized on-policy return; unlike GAE, it has no critic bootstrap.
    mc = np.cumsum(cb.rewards[::-1].astype(np.float64), axis=0)[::-1].copy()
    return dict(mc=mc, prediction=prediction,
                actor_parameters=[parameter_snapshot(a.actor) for a in runner.actor],
                critic_parameters=parameter_snapshot(runner.critic.critic),
                distributions=[actor_distribution(a,b) for a,b in zip(runner.actor, runner.actor_buffer)])


@torch.no_grad()
def after_update(runner, before):
    cb = runner.critic_buffer
    prediction, _ = runner.critic.get_values(
        cb.share_obs[:-1].reshape(-1, cb.share_obs.shape[-1]),
        cb.rnn_states_critic[0], cb.masks[:-1].reshape(-1,1))
    prediction = prediction.cpu().numpy().reshape(cb.rewards.shape)
    if runner.value_normalizer is not None:
        prediction = runner.value_normalizer.denormalize(prediction)
    require_finite('post-update values', prediction)
    result = dict(episode=runner.iteration, sampled_year=runner.sampled_years[-1],
        episode_reward=float(cb.rewards.astype(np.float64).sum()),
        episode_energy_TWh=runner.last_rollout['episode_energy_TWh'],
        critic_update=parameter_update(runner.critic.critic, before['critic_parameters']),
        critic_mc_pre=fit_metrics(before['mc'], before['prediction']),
        critic_mc_post=fit_metrics(before['mc'], prediction),
        critic_gae_post=fit_metrics(cb.returns[:-1], prediction),
        predicted_start_return_TWh=float(before['prediction'][0,0,0]*runner.envs[0].physics.reference_energy/1e6),
        realized_sampled_return_TWh=float(before['mc'][0,0,0]*runner.envs[0].physics.reference_energy/1e6),
        actors=[])
    for i,(actor,buffer,old) in enumerate(zip(runner.actor, runner.actor_buffer,before['distributions'])):
        new = actor_distribution(actor,buffer)
        update = parameter_update(actor.actor, before['actor_parameters'][i])
        active = torch.from_numpy(buffer.active_masks[:-1].reshape(-1)>0)
        if not active.any():
            result['actors'].append(dict(agent=i, active_samples=0, **update)); continue
        action = torch.from_numpy(buffer.actions.reshape(-1,1))
        log_ratio = (new.log_prob(action)-old.log_prob(action))[active]
        ratio = log_ratio.exp()
        kl = torch.distributions.kl_divergence(old,new)[active]
        mean, variance = new.mean[active], new.variance[active]
        require_finite('actor {} ratios'.format(i), ratio.cpu().numpy())
        require_finite('actor {} KL'.format(i), kl.cpu().numpy())
        sampled = action[active]
        result['actors'].append(dict(agent=i, active_samples=int(active.sum()),
            **update,
            entropy=float(new.entropy()[active].mean()),
            sampled_action_mean=float(sampled.mean()),
            sampled_action_std=float(sampled.std(unbiased=False)),
            sampled_boundary_fraction=float(((sampled<=.01)|(sampled>=.99)).float().mean()),
            beta_kl=float(kl.mean().clamp(min=0)),
            ratio_min=float(ratio.min()),ratio_max=float(ratio.max()),
            clip_fraction=float(((ratio<1.-actor.clip_param)|(ratio>1.+actor.clip_param)).float().mean()),
            mean_u=float(mean.mean()),std_u=float(variance.sqrt().mean()),
            eval_mapping_fraction=float((new.mode()[active]**2).mean()),
            expected_sampled_mapping_fraction=float((mean**2+variance).mean())))
    return result


def print_update_window(runner, window=10):
    rows=runner.diagnostics_history[-window:]
    def avg(key,metric):
        values=[r[key][metric] for r in rows if r[key][metric] is not None]
        return float(np.mean(values)) if values else None
    active=[a for r in rows for a in r['actors'] if a['active_samples']]
    summary=dict(episode=runner.iteration,window_episodes=len(rows),
        sampled_years=[r['sampled_year'] for r in rows],
        mc_rmse_pre=avg('critic_mc_pre','rmse'),mc_rmse_post=avg('critic_mc_post','rmse'),
        mc_explained_variance_post=avg('critic_mc_post','explained_variance'),
        gae_explained_variance_post=avg('critic_gae_post','explained_variance'),
        predicted_start_return_TWh=float(np.mean([r['predicted_start_return_TWh'] for r in rows])),
        realized_sampled_return_TWh=float(np.mean([r['realized_sampled_return_TWh'] for r in rows])),
        max_actor_beta_kl=max((a['beta_kl'] for a in active),default=0.),
        mean_actor_clip_fraction=float(np.mean([a['clip_fraction'] for a in active])) if active else 0.,
        ratio_min=min((a['ratio_min'] for a in active),default=1.),
        ratio_max=max((a['ratio_max'] for a in active),default=1.),
        last_actors=rows[-1]['actors'])
    # Existing grad norms are measured BEFORE clipping, not applied update sizes.
    training=runner.training_history[-window:]
    summary['gradient_clip_threshold']=float(runner.algo_args['algo']['max_grad_norm'])
    summary['critic_grad_norm_pre_clip_mean']=float(np.mean([r['critic_critic_grad_norm'] for r in training]))
    summary['last_episode_reward']=rows[-1]['episode_reward']
    summary['last_episode_energy_TWh']=rows[-1]['episode_energy_TWh']
    summary['last_critic_update']=rows[-1]['critic_update']
    summary['last_training_metrics']={k:v for k,v in training[-1].items()
                                     if k.startswith('actor') or k.startswith('critic_')}
    summary['happo_factor_min']=min(r['actor{}_factor_min'.format(i)] for r in training for i in range(5))
    summary['happo_factor_max']=max(r['actor{}_factor_max'.format(i)] for r in training for i in range(5))
    print('DIAG_UPDATE '+json.dumps(summary),flush=True)


def validation_report(runner, deterministic_rows):
    """Compare deployed mean actions with sampled policy, preserving RNG state."""
    repeats=int(runner.algo_args['experiment'].get('diagnostic_stochastic_repeats',3))
    samples=[]
    for repeat in range(repeats):
        values=[]
        for year in runner.data.validation_years:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(1000000+runner.seed*100000+repeat*10000+year)
                row=evaluate_policy(runner.actor,runner.data,runner.physical_config,year,
                    runner.seed,runner.iteration,runner.training_steps,deterministic=False)
            values.append(row['energy_TWh'])
        samples.append(float(np.mean(values)))
    report=dict(episode=runner.iteration,
        deterministic_mean_TWh=float(np.mean([r['energy_TWh'] for r in deterministic_rows])),
        stochastic_mean_TWh=float(np.mean(samples)),
        stochastic_repeat_means_TWh=samples,
        stochastic_repeat_sd_TWh=float(np.std(samples,ddof=1)) if repeats>1 else 0.,
        years=deterministic_rows)
    print('DIAG_VALIDATION '+json.dumps(report),flush=True)
    return report
