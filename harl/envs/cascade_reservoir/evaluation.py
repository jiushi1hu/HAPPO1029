"""Frozen annual evaluation; all metrics stay in memory."""
import numpy as np
import torch
from harl.envs.cascade_reservoir.cascade_reservoir_env import CascadeReservoirEnv, SEASONAL_PHASES
from harl.envs.cascade_reservoir.data_loader import IDS
from harl.envs.cascade_reservoir.reservoir_physics import InfeasibleCascade
from harl.envs.cascade_reservoir.terminal_storage import TerminalReleaseBounds
from harl.envs.cascade_reservoir.reward import EpisodeRewardMetrics


def reference_metadata(physical_config):
    if physical_config.get('seasonal_constraints', False):
        return dict(reference_type='seasonal_target_tracking', reference_label='Seasonal rule reference',
            reference_description='Tracks supplied soft daily end-storage targets on the policy action grid; '
                'uses the same hard seasonal bounds, future-inflow feasibility and terminal storage mode. '
                'A feasible rule controller, not an optimal-generation bound.')
    return dict(reference_type='constant_storage', reference_label='Constant-storage reference',
        reference_description='Maintains initial storage all year with cumulative natural inflow releases; '
            'seasonal operating constraints are disabled.')


def validate_operating_feasibility(data, physical_config, years=None):
    """Preflight physical feasibility only; no policy, reward fit or energy optimization."""
    years = range(data.start_year, data.end_year+1) if years is None else years
    checked = []
    for year in years:
        try:
            env = CascadeReservoirEnv(data, physical_config, start_year=year, end_year=year)
            bounds = {} if env.seasonal is None else dict(
                storage_lower=env.storage_lower, storage_upper=env.storage_upper)
            TerminalReleaseBounds(data, env.physics.dt, env.storage,
                data.external[env.start:env.stop], env.terminal_storage_target, **bounds).check_feasible()
        except RuntimeError as exc:
            raise type(exc)('Operating feasibility check failed for year {}: {}'.format(year, exc)) from exc
        checked.append(int(year))
    return checked


class SeasonalMetrics:
    """Identical compliance accounting for policy and reference trajectories."""

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
def evaluate_policy(actors, data, physical_config, year, seed, iteration, training_steps, deterministic=True, diagnostics=False):
    bins = actors[0].args.get('action_bins', 200)
    env = CascadeReservoirEnv(data, physical_config, start_year=year, end_year=year, action_bins=bins)
    initial = env.storage.copy()
    seasonal_metrics = SeasonalMetrics(env) if env.seasonal is not None else None
    reward_metrics = EpisodeRewardMetrics()
    energy, spill, residual = 0., 0., 0.
    energy_by_reservoir, spill_by_reservoir = np.zeros(5), np.zeros(5)
    action_sum, mapped_sum, active_count = np.zeros(5), np.zeros(5), np.zeros(5)
    head_sum, head_min, head_max = np.zeros(5), np.full(5, np.inf), np.full(5, -np.inf)
    for actor in actors:
        actor.prep_rollout()
    for _ in range(env.stop - env.start):
        obs = env.observation()[None, :]
        actions = []
        for actor in actors:
            action = actor.act(obs, deterministic=deterministic)
            actions.append(float(action.item()))
        env.step(np.asarray(actions))
        r = env.last_transition
        reward_metrics.update(r)
        energy += float(r["energy"].sum())
        spill += float(r["spill"].sum() * env.physics.dt)
        residual = max(residual, float(np.abs(r["residual"]).max()))
        if seasonal_metrics is not None:
            seasonal_metrics.update()
        if diagnostics:
            energy_by_reservoir += r["energy"]
            spill_by_reservoir += r["spill"]*env.physics.dt
            action_sum += r["normalized_actions"]*r["controls"]
            mapped_sum += r["normalized_actions"]**2*r["controls"]
            active_count += r["controls"]
            head_sum += r["net_head_m"]
            head_min = np.minimum(head_min, r["net_head_m"])
            head_max = np.maximum(head_max, r["net_head_m"])
    row = dict(seed=seed, episode=iteration, training_steps=training_steps, year=year,
               phase="validation" if year in data.validation_years else
                     "test" if year in data.test_years else "training_evaluation",
               energy_MWh=energy, energy_TWh=energy / 1e6,
               spill_volume_m3=spill, max_balance_residual_m3=residual,
               head_model=env.physics.head_model, head_data_status=env.physics.head_data_status,
               feasible=True, violation_count=0)
    for i, name in enumerate(IDS):
        row[name + "_start_m3"] = float(initial[i])
        row[name + "_end_m3"] = float(env.storage[i])
    if seasonal_metrics is not None:
        row.update(seasonal_metrics.finish())
    if env.reward_settings['enabled']:
        row.update(reward_metrics.summary(detailed=True), reward_penalties=env.reward_settings)
    if diagnostics:
        row.update(energy_by_reservoir_TWh=(energy_by_reservoir/1e6).tolist(),
            spill_by_reservoir_billion_m3=(spill_by_reservoir/1e9).tolist(),
            end_storage_fraction=((env.storage-data.vmin)/(data.vmax-data.vmin)).tolist(),
            total_storage_depletion_billion_m3=float((initial-env.storage).sum()/1e9),
            mean_u=(action_sum/np.maximum(active_count,1)).tolist(),
            mean_mapping_fraction=(mapped_sum/np.maximum(active_count,1)).tolist(),
            active_days=active_count.astype(int).tolist(),
            mean_net_head_m=(head_sum/(env.stop-env.start)).tolist(),
            min_net_head_m=head_min.tolist(), max_net_head_m=head_max.tolist())
    return row


def reference_summary(data, physics, years, action_bins=200):
    """Reference matched to the selected operating mode and physical model.

    Seasonal targets are soft: low-flow years can deviate from the prescribed
    curve while obeying the same hard constraints as the learned controller.
    The legacy analytic reference is retained only when seasonality is off.
    """
    rows = []
    for year in years:
        if physics.config.get('seasonal_constraints', False):
            env = CascadeReservoirEnv(data, physics.config, start_year=year, end_year=year,
                                      action_bins=action_bins)
            initial = env.storage.copy()
            metrics = SeasonalMetrics(env)
            reward_metrics = EpisodeRewardMetrics()
            energy = np.zeros(5); spill = np.zeros(5); head = np.zeros(5)
            residual = 0.
            while env.index < env.stop:
                env.step_towards_storage(env.seasonal.reference['target_end_of_day_storage_m3'][env.index])
                r = env.last_transition
                reward_metrics.update(r)
                energy += r['energy']; spill += r['spill']*physics.dt; head += r['net_head_m']
                residual = max(residual, float(np.abs(r['residual']).max()))
                metrics.update()
            row = dict(year=year, energy_MWh=float(energy.sum()), energy_TWh=float(energy.sum()/1e6),
                energy_by_reservoir_TWh=(energy/1e6).tolist(), spill_volume_m3=float(spill.sum()),
                spill_by_reservoir_billion_m3=(spill/1e9).tolist(), max_balance_residual_m3=residual,
                mean_net_head_m=(head/(env.stop-env.start)).tolist(), head_model=physics.head_model,
                head_data_status=physics.head_data_status, **reference_metadata(physics.config))
            row.update(metrics.finish())
            if env.reward_settings['enabled']:
                row.update(reward_metrics.summary(detailed=True), reward_penalties=env.reward_settings)
            for i, name in enumerate(IDS):
                row[name+'_start_m3'] = float(initial[i])
                row[name+'_end_m3'] = float(env.storage[i])
            rows.append(row)
            continue
        start, stop = data.indices(year, year)
        _, energy = physics.constant_storage(data.external[start:stop], data.initial_storage(year))
        rows.append(dict(year=year, energy_TWh=float(energy.sum() / 1e6),
                         head_model=physics.head_model, head_data_status=physics.head_data_status))
    return rows
