"""Frozen complete-episode evaluation for the five-reservoir cascade."""
import numpy as np
import torch
from harl.envs.cascade_reservoir.cascade_reservoir_env import SEASONAL_PHASES
from harl.envs.cascade_reservoir.data_loader import IDS
from harl.envs.cascade_reservoir.episodes import (
    DEFAULT_ACTION_BINS, DEFAULT_EPISODE_DAYS, episode_environment)
from harl.envs.cascade_reservoir.reservoir_physics import InfeasibleCascade
from harl.envs.cascade_reservoir.reward import (
    EpisodeRewardMetrics, validate_discount_factor, REWARD_AGGREGATION, reward_metadata)


class SeasonalMetrics:
    """Compliance accounting across every source year in a complete episode."""

    def __init__(self, env):
        self.env = env
        self.days = 0
        self.violations = 0
        self.max_bound_excess = 0.
        self.flood_excess = np.zeros(5)
        self.refill_shortfall = np.zeros(5)
        self.refill_checks = np.zeros(5, dtype=int)
        self.target_error = np.zeros(5)
        self.phase_days = np.zeros((len(SEASONAL_PHASES), 5), dtype=int)
        self.phase_energy = np.zeros((len(SEASONAL_PHASES), 5))

    def update(self):
        env = self.env
        r = env.last_transition
        index = env.index-1
        below = np.maximum(0., r['storage_lower_m3']-r['storage_end'])
        above = np.maximum(0., r['storage_end']-r['storage_upper_m3'])
        excess = np.maximum(below, above)
        tolerance = np.maximum(1., 1e-10*np.abs(r['storage_start']))
        self.violations += int((excess > tolerance).sum())
        self.max_bound_excess = max(self.max_bound_excess, float(excess.max()))
        self.flood_excess = np.maximum(self.flood_excess, above*r['flood_limit_active'])
        self.refill_shortfall = np.maximum(self.refill_shortfall, below*r['refill_deadline'])
        self.refill_checks += r['refill_deadline']
        target = env.seasonal.reference['target_end_of_day_level_m'][index]
        self.target_error += np.abs(r['levels_end']-target)
        for i, phase in enumerate(SEASONAL_PHASES):
            mask = np.asarray(r['seasonal_phases']) == phase
            self.phase_days[i] += mask
            self.phase_energy[i] += r['energy']*mask
        self.days += 1

    def finish(self):
        env = self.env
        closure = (np.zeros(5) if env.terminal_storage_target is None else
                   np.abs(env.storage-env.terminal_storage_target))
        tolerance = np.maximum(1., 1e-10*np.abs(env.storage))
        violations = self.violations + int((closure > tolerance).sum())
        if violations:
            raise InfeasibleCascade('Evaluation trajectory violates operating constraints')
        return dict(seasonal_constraints=True, action_bins=env.action_bins,
            operation_data_status={name: env.seasonal.key_parameters[name]['value_status'] for name in IDS},
            feasible=True, violation_count=violations,
            max_daily_bound_excess_m3=self.max_bound_excess,
            flood_ceiling_excess_m3_by_reservoir=self.flood_excess.tolist(),
            refill_deadline_shortfall_m3_by_reservoir=self.refill_shortfall.tolist(),
            refill_deadline_checks_by_reservoir=self.refill_checks.tolist(),
            terminal_storage_mode=env.terminal_storage_mode,
            terminal_storage_error_m3_by_reservoir=closure.tolist(),
            soft_reference_level_mae_m_by_reservoir=(self.target_error/self.days).tolist(),
            phase_days_by_reservoir={p: self.phase_days[i].tolist() for i,p in enumerate(SEASONAL_PHASES)},
            phase_energy_by_reservoir_TWh={p: (self.phase_energy[i]/1e6).tolist()
                                          for i,p in enumerate(SEASONAL_PHASES)})


@torch.no_grad()
def evaluate_policy(actors, data, physical_config, seed, episode, training_steps, *,
                    gamma, source_years, episode_days=DEFAULT_EPISODE_DAYS,
                    deterministic=True, diagnostics=False):
    """Evaluate one complete multi-year sequence without collecting training samples.

    Evaluation years are used only after training. A training-only diagnostic
    may explicitly supply training years; mixed split sequences are rejected.
    Source dates and seasonal rules are preserved at representative-year joins.
    """
    gamma = validate_discount_factor(gamma)
    if len(actors) != len(IDS):
        raise ValueError('Evaluation requires one actor per reservoir')
    bins = actors[0].args.get('action_bins', DEFAULT_ACTION_BINS)
    if any(actor.args.get('action_bins', DEFAULT_ACTION_BINS) != bins for actor in actors):
        raise ValueError('Evaluation actors must use one common discrete action grid')
    env = episode_environment(data, physical_config, source_years, episode_days, bins)
    used_years = list(dict.fromkeys(day.year for day in env.data.dates[env.start:env.stop]))
    if set(used_years).issubset(data.evaluation_years):
        phase = 'evaluation'
    elif set(used_years).issubset(data.training_years):
        phase = 'training_evaluation'
    else:
        raise ValueError('An evaluation episode cannot mix training and evaluation years')
    initial = env.storage.copy()
    seasonal_metrics = SeasonalMetrics(env) if env.seasonal is not None else None
    reward_metrics = EpisodeRewardMetrics(gamma)
    energy, spill, residual = 0., 0., 0.
    energy_by_reservoir, spill_by_reservoir = np.zeros(5), np.zeros(5)
    action_sum, mapped_sum, active_count = np.zeros(5), np.zeros(5), np.zeros(5)
    head_sum, head_min, head_max = np.zeros(5), np.full(5, np.inf), np.full(5, -np.inf)
    for actor in actors:
        actor.prep_rollout()
    for _ in range(env.stop-env.start):
        obs = env.actor_observations()
        actions = [float(actor.act(obs[i:i+1], deterministic=deterministic).item())
                   for i, actor in enumerate(actors)]
        env.step(np.asarray(actions))
        row = env.last_transition
        reward_metrics.update(row)
        energy += float(row['energy'].sum())
        spill += float(row['spill'].sum()*env.physics.dt)
        residual = max(residual, float(np.abs(row['residual']).max()))
        if seasonal_metrics is not None:
            seasonal_metrics.update()
        if diagnostics:
            energy_by_reservoir += row['energy']
            spill_by_reservoir += row['spill']*env.physics.dt
            action_sum += row['normalized_actions']*row['controls']
            mapped_sum += row['normalized_actions']**2*row['controls']
            active_count += row['controls']
            head_sum += row['net_head_m']
            head_min = np.minimum(head_min, row['net_head_m'])
            head_max = np.maximum(head_max, row['net_head_m'])
    closure = (np.zeros(5) if env.terminal_storage_target is None else
               np.abs(env.storage-env.terminal_storage_target))
    tolerance = np.maximum(1., 1e-10*np.abs(env.storage))
    if (closure > tolerance).any():
        raise InfeasibleCascade('Evaluation terminal storage target was not reached')
    result = dict(seed=int(seed), episode=int(episode), training_steps=int(training_steps),
        phase=phase, source_years=used_years, window_start_year=used_years[0],
        episode_horizon_days=env.stop-env.start,
        start_date=env.data.dates[env.start].isoformat(),
        end_date=env.data.dates[env.stop-1].isoformat(),
        source_sequence='ordered source years; storage carries across every boundary; leap days retained',
        energy_MWh=energy, energy_TWh=energy/1e6, spill_volume_m3=spill,
        max_balance_residual_m3=residual, head_model=env.physics.head_model,
        head_data_status=env.physics.head_data_status, terminal_storage_mode=env.terminal_storage_mode,
        terminal_storage_error_m3_by_reservoir=closure.tolist(),
        feasible=True, violation_count=0, **env.observation_metadata())
    for i, name in enumerate(IDS):
        result[name+'_start_m3'] = float(initial[i])
        result[name+'_end_m3'] = float(env.storage[i])
    if seasonal_metrics is not None:
        result.update(seasonal_metrics.finish())
    result.update(reward_metrics.summary(detailed=True), **reward_metadata(data),
        reward_penalties=env.reward_settings, reward_aggregation=REWARD_AGGREGATION)
    if diagnostics:
        result.update(energy_by_reservoir_TWh=(energy_by_reservoir/1e6).tolist(),
            spill_by_reservoir_billion_m3=(spill_by_reservoir/1e9).tolist(),
            end_storage_fraction=((env.storage-data.vmin)/(data.vmax-data.vmin)).tolist(),
            total_storage_depletion_billion_m3=float((initial-env.storage).sum()/1e9),
            mean_u=(action_sum/np.maximum(active_count, 1)).tolist(),
            mean_mapping_fraction=(mapped_sum/np.maximum(active_count, 1)).tolist(),
            active_days=active_count.astype(int).tolist(),
            mean_net_head_m=(head_sum/(env.stop-env.start)).tolist(),
            min_net_head_m=head_min.tolist(), max_net_head_m=head_max.tolist())
    return result
