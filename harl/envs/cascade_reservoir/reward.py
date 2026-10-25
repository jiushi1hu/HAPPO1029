"""Dimensionless soft-operation penalties and shared episode accounting."""
import numpy as np
from harl.envs.cascade_reservoir.data_loader import IDS

PENALTY_TERMS = ('guaranteed_output', 'ecological_flow')
REWARD_DEFINITION = 'generation_minus_mean_service_shortfalls_v1'
SHORTFALL_UNITS = dict(guaranteed_output='MW', ecological_flow='m3/s')
INTEGRAL_FIELDS = dict(guaranteed_output='shortfall_energy_MWh', ecological_flow='deficit_volume_m3')
REWARD_AGGREGATION = 'sum(gamma**t * daily_reward), t=0 at episode start'


def reward_metadata(data):
    requirements = getattr(data, 'service_requirements', None)
    return dict(reward_definition=REWARD_DEFINITION,
                service_requirements=None if requirements is None else requirements.metadata())


def validate_discount_factor(value):
    """Validate the one discount used by training, evaluation and accounting."""
    message = "Discount factor gamma must be finite and between 0 and 1"
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    try:
        gamma = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(gamma) or not 0. <= gamma <= 1.:
        raise ValueError(message)
    return gamma


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
        value = raw.get(key, 1.)
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(key+' must be finite and nonnegative')
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(key+' must be finite and nonnegative') from exc
        if not np.isfinite(value) or value < 0:
            raise ValueError(key+' must be finite and nonnegative')
        settings[key] = value
    return settings


class OperationReward:
    """One team reward: generation minus mean relative service shortfalls.

    Each denominator is a fixed positive requirement. Zero disables that
    reservoir's requirement; means use the fixed active set, never just the
    violating reservoirs. Seasonality does not gate these service penalties.
    """

    def __init__(self, data, settings, timestep_seconds=86400.):
        self.settings = dict(settings)
        self.requirements = getattr(data, 'service_requirements', None)
        if self.requirements is None and settings['enabled']:
            loader = getattr(data, 'load_service_requirements', None)
            if loader is None:
                raise ValueError('Reward penalties require service requirement data')
            self.requirements = loader()
        self.dt = float(timestep_seconds)
        if not np.isfinite(self.dt) or self.dt <= 0.:
            raise ValueError('Service shortfall accounting requires a positive time step')
        if self.requirements is not None:
            self.thresholds = dict(guaranteed_output=np.array(self.requirements.guaranteed_output_mw),
                                   ecological_flow=np.array(self.requirements.ecological_flow_m3s))
            if any(v.shape != (len(IDS),) or not np.isfinite(v).all() or (v < 0.).any()
                   for v in self.thresholds.values()):
                raise ValueError('Service requirements must be finite nonnegative five-vectors')

    def apply(self, transition):
        if self.requirements is None:
            return
        actual = dict(guaranteed_output=np.asarray(transition['power']),
                      ecological_flow=np.asarray(transition['release']))
        if any(v.shape != (len(IDS),) or not np.isfinite(v).all() or (v < 0.).any()
               for v in actual.values()):
            raise ValueError('Service reward requires actual finite nonnegative power and release')
        active = {k: threshold > 0. for k, threshold in self.thresholds.items()}
        excess, normalized, by_reservoir = {}, {}, {}
        for k, threshold in self.thresholds.items():
            raw = np.maximum(threshold-actual[k], 0.)
            # Ignore only relative floating-point roundoff, in costs and counts.
            excess[k] = np.where(raw > 1e-10*np.maximum(threshold, 1.), raw, 0.)
            normalized[k] = np.divide(excess[k], threshold, out=np.zeros(len(IDS)), where=active[k])
            weight = self.settings[k+'_weight'] if self.settings['enabled'] else 0.
            by_reservoir[k] = weight*normalized[k]/max(int(active[k].sum()), 1)
        costs = {k: float(v.sum()) for k, v in by_reservoir.items()}
        penalty = float(sum(costs.values()))
        generation_reward = transition.get('generation_reward', transition['reward'])
        reward = generation_reward-penalty
        if not np.isfinite(reward):
            raise FloatingPointError('Non-finite penalized reward')
        transition.update(generation_reward=generation_reward, reward=reward, reward_penalty=penalty,
            reward_penalty_components=costs, service_shortfalls=excess,
            normalized_service_shortfalls=normalized, service_active_masks=active,
            service_shortfall_integrals=dict(guaranteed_output=excess['guaranteed_output']*self.dt/3600.,
                                            ecological_flow=excess['ecological_flow']*self.dt),
            reward_penalty_by_reservoir=sum(by_reservoir.values()))


class EpisodeRewardMetrics:
    """Sum gamma**t times each reward/cost, with t=0 at episode start.

    Daily rewards passed to GAE remain unchanged. Physical exceedance counts
    and the internal energy reconciliation are never discounted.
    """

    def __init__(self, gamma):
        self.gamma = validate_discount_factor(gamma)
        self.discount = 1.
        self.steps = 0
        self.reward = self.generation_reward = self.penalty = 0.
        self.undiscounted_generation_reward = 0.
        self.components = dict.fromkeys(PENALTY_TERMS,0.)
        self.by_reservoir = np.zeros(len(IDS))
        self.violation_days = {k: np.zeros(len(IDS),dtype=int) for k in PENALTY_TERMS}
        self.violation_max = {k: np.zeros(len(IDS)) for k in PENALTY_TERMS}
        self.violation_any_days = dict.fromkeys(PENALTY_TERMS, 0)
        self.active_days = {k: np.zeros(len(IDS), dtype=int) for k in PENALTY_TERMS}
        self.relative_sum = {k: np.zeros(len(IDS)) for k in PENALTY_TERMS}
        self.relative_max = {k: np.zeros(len(IDS)) for k in PENALTY_TERMS}
        self.integrals = {k: np.zeros(len(IDS)) for k in PENALTY_TERMS}

    def update(self, transition):
        generation = float(transition.get('generation_reward',transition['reward']))
        self.reward += self.discount * float(transition['reward'])
        self.generation_reward += self.discount * generation
        self.undiscounted_generation_reward += generation
        self.penalty += self.discount * float(transition.get('reward_penalty',0.))
        self.by_reservoir += self.discount * transition.get('reward_penalty_by_reservoir',0.)
        for term in PENALTY_TERMS:
            self.components[term] += self.discount * transition.get('reward_penalty_components',{}).get(term,0.)
            excess = transition.get('service_shortfalls',{}).get(term,np.zeros(len(IDS)))
            relative = transition.get('normalized_service_shortfalls',{}).get(term,np.zeros(len(IDS)))
            self.active_days[term] += transition.get('service_active_masks',{}).get(term,False)
            self.relative_sum[term] += relative
            self.relative_max[term] = np.maximum(self.relative_max[term], relative)
            self.integrals[term] += transition.get('service_shortfall_integrals',{}).get(term,0.)
            self.violation_days[term] += excess > 0.
            self.violation_max[term] = np.maximum(self.violation_max[term], excess)
            self.violation_any_days[term] += int(np.any(excess > 0.))
        self.steps += 1
        self.discount *= self.gamma

    def summary(self, detailed=False):
        result = dict(gamma=self.gamma, episode_reward=self.reward, episode_generation_reward=self.generation_reward,
                      episode_penalty=self.penalty)
        result.update({'episode_penalty_'+k:v for k,v in self.components.items()})
        if detailed:
            result.update(penalty_by_reservoir=self.by_reservoir.tolist(),
                          service_shortfalls=self.shortfall_summary())
        return result

    def shortfall_summary(self):
        """Undiscounted service reliability, severity, and physical deficits."""
        result = {}
        for term in PENALTY_TERMS:
            days, active = self.violation_days[term], self.active_days[term]
            count, eligible = int(days.sum()), int(active.sum())
            integral = INTEGRAL_FIELDS[term]
            result[term] = dict(reservoir_day_count=count, active_reservoir_days=eligible,
                reservoir_day_rate=count/eligible if eligible else None,
                compliance_rate=1.-count/eligible if eligible else None,
                days_any_reservoir=int(self.violation_any_days[term]),
                shortfall_unit=SHORTFALL_UNITS[term], max_shortfall=float(self.violation_max[term].max()),
                mean_relative_shortfall=float(self.relative_sum[term].sum())/eligible if eligible else None,
                max_relative_shortfall=float(self.relative_max[term].max()),
                by_reservoir_days=days.tolist(), by_reservoir_active_days=active.tolist(),
                by_reservoir_compliance_rate=[1.-int(d)/int(a) if a else None for d,a in zip(days,active)],
                by_reservoir_max_shortfall=self.violation_max[term].tolist(),
                **{integral:float(self.integrals[term].sum()),
                   'by_reservoir_'+integral:self.integrals[term].tolist()})
        return result
