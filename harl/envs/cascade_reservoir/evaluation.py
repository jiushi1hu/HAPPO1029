"""Frozen annual evaluation; all metrics stay in memory."""
import numpy as np
import torch
from harl.envs.cascade_reservoir.cascade_reservoir_env import CascadeReservoirEnv
from harl.envs.cascade_reservoir.data_loader import IDS


@torch.no_grad()
def evaluate_policy(actors, data, physical_config, year, seed, iteration, training_steps, deterministic=True, diagnostics=False):
    bins = actors[0].args.get('action_bins', 200)
    env = CascadeReservoirEnv(data, physical_config, start_year=year, end_year=year, action_bins=bins)
    initial = env.storage.copy()
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
        energy += float(r["energy"].sum())
        spill += float(r["spill"].sum() * env.physics.dt)
        residual = max(residual, float(np.abs(r["residual"]).max()))
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


def reference_summary(data, physics, years):
    rows = []
    for year in years:
        start, stop = data.indices(year, year)
        _, energy = physics.constant_storage(data.external[start:stop], data.initial_storage(year))
        rows.append(dict(year=year, energy_TWh=float(energy.sum() / 1e6),
                         head_model=physics.head_model, head_data_status=physics.head_data_status))
    return rows
