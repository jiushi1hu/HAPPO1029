"""Read-only accounting of sampled episodes, separate from rewards and updates."""
import json
import numpy as np
from harl.envs.cascade_reservoir.data_loader import IDS
from harl.envs.cascade_reservoir.reward import PENALTY_TERMS, REWARD_AGGREGATION, REWARD_DEFINITION


def episode_record(env, rewards, energy_by_reservoir, seed, episode, gamma,
                   gae_lambda, failed_episodes, failure=None):
    """Describe real executed days; a rejected transition contributes no energy.

    Failure rates count failed episode attempts, not soft reservoir-day events.
    The current runner still stops on a hard failure after emitting this record.
    """
    days = env.index-env.start
    completed = failure is None and env.index == env.stop
    failed = failure is not None
    if not (completed or failed):
        raise ValueError('Episode accounting requires completion or an explicit failure')
    energy = np.asarray(energy_by_reservoir, dtype=np.float64)
    if energy.shape != (5,) or not np.isfinite(energy).all() or (energy < 0.).any():
        raise ValueError('Generation must be a finite nonnegative five-vector')
    totals = rewards.summary()
    if rewards.gamma != gamma or rewards.steps != days:
        raise ValueError('Episode accounting must use the training gamma and executed days')
    power = totals['episode_generation_reward']
    costs = {k: totals['episode_penalty_'+k] for k in PENALTY_TERMS}
    if not np.isclose(rewards.undiscounted_generation_reward,
                      energy.sum()/env.physics.reference_energy, rtol=1e-10, atol=1e-9):
        raise ValueError('Power Reward and actual generation disagree')
    if not np.isclose(totals['episode_reward'], power-sum(costs.values()), rtol=1e-10, atol=1e-9):
        raise ValueError('Reward components do not sum to Total Reward')
    percentages = {k: 100.*v/power if power > 0. else None for k,v in costs.items()}
    percentages['total'] = 100.*sum(costs.values())/power if power > 0. else None
    requirements = getattr(env.data, 'service_requirements', None)
    return dict(record_type='training_episode', seed=int(seed), episode=int(episode),
        window_start_year=int(env.data.dates[env.start].year), completed=completed,
        source_years=list(dict.fromkeys(day.year for day in env.data.dates[env.start:env.stop])),
        episode_horizon_days=int(env.stop-env.start),
        start_date=env.data.dates[env.start].isoformat(),
        end_date=env.data.dates[env.stop-1].isoformat(),
        calendar_days=int(env.stop-env.start), simulated_days=int(days),
        actual_generation_MWh=float(energy.sum()), actual_generation_TWh=float(energy.sum()/1e6),
        episode_generation_TWh=float(energy.sum()/1e6) if completed else None,
        generation_MWh_by_reservoir=energy.tolist(), reservoir_ids=list(IDS),
        power_reward=power, **{k+'_penalty':v for k,v in costs.items()},
        total_soft_penalty=sum(costs.values()),
        total_reward=totals['episode_reward'], penalty_to_power_percent=percentages,
        hard_constraint_failed=failed, hard_constraint_failure_rate=float(failed),
        hard_constraint_failure_rate_so_far=failed_episodes/float(episode),
        attempted_episodes=int(episode), failed_episodes_so_far=int(failed_episodes), failure=failure,
        service_requirements_available=requirements is not None,
        service_shortfalls=rewards.shortfall_summary(),
        service_requirements=None if requirements is None else requirements.metadata(),
        gamma=float(gamma), gae_lambda=float(gae_lambda),
        reward_aggregation=REWARD_AGGREGATION, reward_definition=REWARD_DEFINITION,
        **env.observation_metadata(),
        reward_penalties=dict(env.reward_settings), reference_energy_MWh=env.physics.reference_energy,
        feasibility_uses_actual_future_inflows=bool(env.seasonal is not None or
                                                   env.terminal_storage_target is not None))


def log_episode(record, stream=None):
    """Always print JSON; optionally flush one JSONL record to an explicit file."""
    encoded = json.dumps(record, allow_nan=False)
    if stream is not None:
        stream.write(encoded+'\n')
        stream.flush()
    print('EPISODE_METRICS '+encoded, flush=True)
