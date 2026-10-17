"""Dimensionless soft-operation penalties and shared episode accounting."""
import numpy as np
from harl.envs.cascade_reservoir.data_loader import IDS, OPERATION_PERIODS

PENALTY_TERMS = ('target_band', 'daily_rise', 'daily_fall')


def reward_settings(config):
    """Validate effective settings before loading data or running simulations."""
    raw = config.get('reward_penalties', {})
    if not isinstance(raw, dict):
        raise ValueError('reward_penalties must be a mapping')
    allowed = {'enabled'} | {term+'_weight' for term in PENALTY_TERMS}
    if set(raw)-allowed:
        raise ValueError('Unknown reward_penalties settings: '+str(sorted(map(str,set(raw)-allowed))))
    enabled = raw.get('enabled', False)
    if not isinstance(enabled, bool):
        raise ValueError('reward_penalties.enabled must be a boolean')
    settings = dict(enabled=enabled)
    for term in PENALTY_TERMS:
        key = term+'_weight'
        value = raw.get(key, .1)
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(key+' must be finite and nonnegative')
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(key+' must be finite and nonnegative') from exc
        if not np.isfinite(value) or value < 0:
            raise ValueError(key+' must be finite and nonnegative')
        settings[key] = value
    if enabled and not config.get('seasonal_constraints', False):
        raise ValueError('Reward penalties require seasonal_constraints=true')
    return settings


class OperationReward:
    """Penalize soft deviations without relaxing physical feasibility.

    Each excess is measured in metres, divided by that reservoir's global
    operating level range, averaged over five reservoirs, then weighted. The
    weights are scenario choices, not monetary costs or calibrated estimates.
    """

    def __init__(self, data, seasonal, settings):
        self.data, self.seasonal, self.settings = data, seasonal, dict(settings)
        if not settings['enabled']:
            return
        if seasonal is None:
            raise ValueError('Reward penalties require seasonal operating data')
        self.level_range = data.levels(data.vmax)-data.levels(data.vmin)
        if not np.isfinite(self.level_range).all() or (self.level_range <= 0).any():
            raise ValueError('Penalty normalization requires positive operating level ranges')
        self.rate_limits = {}
        for period in OPERATION_PERIODS:
            self.rate_limits[period] = tuple(np.array([
                seasonal.period_parameters[(name,period)][field] for name in IDS], dtype=float)
                for field in ('suggested_soft_daily_rise_limit_m', 'suggested_soft_daily_fall_limit_m'))
        lower = seasonal.reference['soft_target_lower_level_m']
        upper = seasonal.reference['soft_target_upper_level_m']
        if (not np.isfinite(lower).all() or not np.isfinite(upper).all() or (lower > upper).any()):
            raise ValueError('Soft target bands must be finite and ordered')

    def apply(self, transition, index):
        if not self.settings['enabled']:
            return
        level = transition['levels_end']
        delta = level-self.data.levels(transition['storage_start'])
        rise, fall = self.rate_limits[self.data.period_labels[index]]
        reference = self.seasonal.reference
        excess = dict(target_band=np.maximum(reference['soft_target_lower_level_m'][index]-level,0.)+
                                  np.maximum(level-reference['soft_target_upper_level_m'][index],0.),
                      daily_rise=np.maximum(delta-rise,0.), daily_fall=np.maximum(-delta-fall,0.))
        # Ignore interpolation roundoff below one nanometre, consistently in
        # reward calculations and reported soft-deviation counts.
        excess = {k: np.where(v > 1e-9,v,0.) for k,v in excess.items()}
        normalized = {k: v/self.level_range for k,v in excess.items()}
        costs = {k: self.settings[k+'_weight']*float(v.mean()) for k,v in normalized.items()}
        penalty = float(sum(costs.values()))
        generation_reward = transition['reward']
        reward = generation_reward-penalty
        if not np.isfinite(reward):
            raise FloatingPointError('Non-finite penalized reward')
        transition.update(generation_reward=generation_reward, reward=reward, reward_penalty=penalty,
            reward_penalty_components=costs, soft_violations_m=excess,
            normalized_soft_violations=normalized,
            reward_penalty_by_reservoir=sum(self.settings[k+'_weight']*v/len(IDS)
                                           for k,v in normalized.items()))


class EpisodeRewardMetrics:
    """Use the same reward/cost totals in training, evaluation and references."""

    def __init__(self):
        self.reward = self.generation_reward = self.penalty = 0.
        self.components = dict.fromkeys(PENALTY_TERMS,0.)
        self.by_reservoir = np.zeros(len(IDS))
        self.violation_days = {k: np.zeros(len(IDS),dtype=int) for k in PENALTY_TERMS}
        self.violation_sum = {k: np.zeros(len(IDS)) for k in PENALTY_TERMS}

    def update(self, transition):
        self.reward += float(transition['reward'])
        self.generation_reward += float(transition.get('generation_reward',transition['reward']))
        self.penalty += float(transition.get('reward_penalty',0.))
        self.by_reservoir += transition.get('reward_penalty_by_reservoir',0.)
        for term in PENALTY_TERMS:
            self.components[term] += transition.get('reward_penalty_components',{}).get(term,0.)
            excess = transition.get('soft_violations_m',{}).get(term,np.zeros(5))
            self.violation_days[term] += excess > 0.
            self.violation_sum[term] += excess

    def summary(self, detailed=False):
        result = dict(episode_reward=self.reward, episode_generation_reward=self.generation_reward,
                      episode_penalty=self.penalty)
        result.update({'episode_penalty_'+k:v for k,v in self.components.items()})
        if detailed:
            result.update(penalty_by_reservoir=self.by_reservoir.tolist(),
                soft_violation_days_by_reservoir={k:v.tolist() for k,v in self.violation_days.items()},
                soft_violation_sum_m_by_reservoir={k:v.tolist() for k,v in self.violation_sum.items()})
        return result
