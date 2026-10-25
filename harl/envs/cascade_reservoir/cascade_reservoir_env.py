"""Local actor observations and a centralized state for the reservoir cascade."""
import calendar
import numpy as np
from gym.spaces import Box, Discrete
from harl.envs.cascade_reservoir.reservoir_physics import ReservoirPhysics, InfeasibleCascade
from harl.envs.cascade_reservoir.terminal_storage import TerminalReleaseBounds
from harl.envs.cascade_reservoir.reward import OperationReward, reward_settings

SEASONAL_PHASES = ("drawdown", "flood_control", "impoundment", "post_refill")
ACTOR_OBSERVATION_LAYOUT = "local_neighbors_v1"
LOCAL_NEIGHBOR_FEATURES = (
    "storage_fraction", "external_inflow_standardized", "previous_release_fraction",
    "upstream_storage_fraction", "upstream_external_inflow_standardized",
    "upstream_previous_release_fraction", "downstream_storage_fraction",
    "downstream_headroom_fraction",
)
LOCAL_RULE_FEATURES = (
    "storage_lower_fraction", "storage_upper_fraction", "deadline_days_remaining",
    "deadline_storage_lower_fraction", "deadline_storage_upper_fraction",
)
TIME_FEATURES = ("season_sin", "season_cos", "episode_days_remaining")


def actor_observation_features(seasonal_constraints):
    return LOCAL_NEIGHBOR_FEATURES + (LOCAL_RULE_FEATURES if seasonal_constraints else ()) + TIME_FEATURES


class CascadeReservoirEnv:
    n_agents = 5
    observation_layout = ACTOR_OBSERVATION_LAYOUT

    def observation_metadata(self):
        return dict(actor_observation_layout=self.observation_layout,
                    actor_observation_size=self.observation_size, critic_state_size=self.state_size)

    def __init__(self, data, config, end_year=None, start_year=None, action_bins=200):
        self.data = data
        self.physics = ReservoirPhysics(data, config)
        self.seasonal_constraints = config.get('seasonal_constraints', False)
        if not isinstance(self.seasonal_constraints, bool):
            raise ValueError('seasonal_constraints must be a boolean')
        self.reward_settings = reward_settings(config)
        self.seasonal = data.load_seasonal_operation() if self.seasonal_constraints else None
        self.operation_reward = OperationReward(data, self.reward_settings,
                                               timestep_seconds=self.physics.dt)
        self.terminal_storage_mode = config.get('terminal_storage_mode', 'free')
        if self.terminal_storage_mode not in ('free', 'initial'):
            raise ValueError('Terminal storage mode must be free or initial')
        start_year = data.start_year if start_year is None else int(start_year)
        end_year = start_year if end_year is None else int(end_year)
        if self.terminal_storage_mode == 'initial' and start_year != end_year:
            raise ValueError('Return-to-initial storage requires a single-year episode')
        self.start, self.stop = data.indices(start_year, end_year)
        if self.seasonal is not None:
            self._prepare_seasonal_calendar()
        # A finite-horizon policy needs time-to-go, not just day of year.
        # Keep the scale fixed across training, validation and test horizons.
        self.time_scale_days = 366  # Fixed annual scale, including leap days.
        self.observation_features = actor_observation_features(self.seasonal_constraints)
        self.observation_size = len(self.observation_features)
        self.state_size = 43 if self.seasonal_constraints else 18
        self.observation_space = [Box(-np.inf, np.inf, (self.observation_size,), dtype=np.float32) for _ in range(5)]
        self.share_observation_space = [Box(-np.inf, np.inf, (self.state_size,), dtype=np.float32) for _ in range(5)]
        if (isinstance(action_bins, bool) or
                not isinstance(action_bins, (int, np.integer)) or action_bins < 2):
            raise ValueError('action_bins must be an integer >= 2')
        self.action_bins = action_bins
        self.action_space = [Discrete(action_bins) for _ in range(5)]
        self.reset()

    def _prepare_seasonal_calendar(self):
        """Cache bounds, per-reservoir phases and hard events for this episode."""
        self.storage_lower, self.storage_upper = self.seasonal.storage_bounds(self.start, self.stop)
        self._phase_codes = np.empty((self.stop-self.start, 5), dtype=np.int8)
        self._deadline_indices = [[] for _ in range(5)]
        for year in range(self.data.dates[self.start].year, self.data.dates[self.stop-1].year+1):
            a, b = self.data.indices(year, year)
            days = np.arange(b-a)
            for j in range(5):
                flood_days = np.flatnonzero(self.seasonal.flood_limit_active[a:b,j])
                refill = int(np.flatnonzero(self.seasonal.refill_deadline[a:b,j])[0])
                draw, end = int(flood_days[0]), int(flood_days[-1])
                codes = np.select([days <= draw, days <= end, days <= refill], [0, 1, 2], default=3)
                self._phase_codes[a-self.start:b-self.start,j] = codes
                self._deadline_indices[j].extend((a+draw, a+refill))
        if self.terminal_storage_mode == 'initial':
            for events in self._deadline_indices:
                events.append(self.stop-1)
        self._deadline_indices = [np.unique(events) for events in self._deadline_indices]

    def _check_previous_storage(self, storage, index):
        """The prior day's constraint applies to today's starting state."""
        if storage.shape != (5,) or not np.isfinite(storage).all():
            raise InfeasibleCascade('Starting seasonal storage must be a finite five-vector')
        tolerance = np.maximum(1., 1e-10*np.abs(storage))
        if index == 0:
            lower, upper = self.data.vmin, self.data.vmax
        else:
            lower = self.seasonal.storage_lower[index-1]
            upper = self.seasonal.storage_upper[index-1]
        if ((storage < lower-tolerance) | (storage > upper+tolerance)).any():
            raise InfeasibleCascade('Starting storage violates previous-day seasonal bounds')

    def reset(self):
        storage = self.data.initial_storage(self.data.dates[self.start].year)
        if self.seasonal is not None:
            self._check_previous_storage(storage, self.start)
            if self.terminal_storage_mode == 'initial' and (
                    (storage < self.storage_lower[-1]) | (storage > self.storage_upper[-1])).any():
                raise InfeasibleCascade('Initial storage target conflicts with final seasonal bounds')
        self.index = self.start
        self.storage = storage
        self.terminal_storage_target = self.storage.copy() if self.terminal_storage_mode == 'initial' else None
        self.previous_release = np.zeros(5)
        self.last_transition = None
        return self._observations()

    def state(self):
        """Centralized critic input; never passed directly to reservoir actors."""
        if self.index == self.stop:
            return np.zeros(self.state_size, dtype=np.float32)
        day = self.data.dates[self.index]
        phase = 2. * np.pi * (day.timetuple().tm_yday - 1) / (366 if calendar.isleap(day.year) else 365)
        parts = [
            (self.storage - self.data.vmin) / (self.data.vmax - self.data.vmin),
            (self.data.external[self.index] - self.data.flow_mean) / self.data.flow_std,
            self.previous_release / self.data.release_cap,
            [np.sin(phase), np.cos(phase)],
            [(self.stop - self.index) / self.time_scale_days],
        ]
        if self.seasonal is not None:
            # Append 25 known operating-rule features, never future inflows or
            # prescribed reference trajectories. Layout: current lower/upper
            # fractions, next hard-deadline time-to-go, next lower/upper fractions.
            offset = self.index-self.start
            span = self.data.vmax-self.data.vmin
            deadline_days = np.full(5, -1.)  # No remaining hard event in this episode.
            deadline_lower, deadline_upper = np.zeros(5), np.ones(5)
            for j, events in enumerate(self._deadline_indices):
                k = int(np.searchsorted(events, self.index))
                if k == len(events):
                    continue
                event = int(events[k]);t = event-self.start
                deadline_days[j] = (event-self.index+1)/self.time_scale_days
                lower, upper = self.storage_lower[t,j], self.storage_upper[t,j]
                if event == self.stop-1 and self.terminal_storage_target is not None:
                    lower = upper = self.terminal_storage_target[j]
                deadline_lower[j] = (lower-self.data.vmin[j])/span[j]
                deadline_upper[j] = (upper-self.data.vmin[j])/span[j]
            parts.extend([(self.storage_lower[offset]-self.data.vmin)/span,
                          (self.storage_upper[offset]-self.data.vmin)/span,
                          deadline_days, deadline_lower, deadline_upper])
        return np.concatenate(parts).astype(np.float32)

    def actor_observations(self):
        """Own state/rules, immediate neighbors, and known calendar timing only.

        All actors decide before today's upstream releases are resolved, so the
        upstream release is yesterday's executed release. External inflows use
        the existing current-day-known forcing convention, not future samples.
        Headroom is signed room below today's end-storage ceiling: it can be
        negative on a drawdown deadline. No system aggregates enter this input.
        """
        if self.index == self.stop:
            return np.zeros((5, self.observation_size), dtype=np.float32)
        state = self.state()
        storage, external, release = state[:5], state[5:10], state[10:15]
        upper = state[23:28] if self.seasonal is not None else np.ones(5)
        observations = []
        for i in range(5):
            upstream = (storage[i-1], external[i-1], release[i-1]) if i else (0., 0., 0.)
            downstream = ((storage[i+1], upper[i+1]-storage[i+1]) if i < 4 else (0., 1.))
            local = [storage[i], external[i], release[i], *upstream, *downstream]
            if self.seasonal is not None:
                local.extend(state[start+i] for start in (18, 23, 28, 33, 38))
            local.extend(state[15:18])
            observations.append(local)
        return np.asarray(observations, dtype=np.float32)

    def _observations(self):
        state = np.repeat(self.state()[None, :], 5, axis=0)
        return self.actor_observations(), state, None

    def normalize_actions(self, actions):
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape not in ((5,), (5,1)) or not np.isfinite(actions).all():
            raise ValueError('Five finite scalar actions required')
        actions = actions.reshape(5)
        if ((actions != np.floor(actions)) | (actions < 0) | (actions >= self.action_bins)).any():
            raise ValueError('Discrete actions must be integer indices in [0, action_bins-1]')
        return actions / (self.action_bins-1)

    def actions_from_releases(self, releases, low, high):
        """Invert given feasible bounds; round to nearest normalized grid index.

        Off-grid releases are quantized in u, not in m3/s. A collapsed interval
        has canonical index zero.
        """
        releases, low, high = [np.asarray(x, dtype=np.float64) for x in (releases, low, high)]
        if any(x.shape != (5,) or not np.isfinite(x).all() for x in (releases, low, high)):
            raise ValueError('Expected finite five-vectors for release bounds')
        if ((high < low) | (releases < low) | (releases > high)).any():
            raise ValueError('Release outside feasible bounds')
        active = high-low > 1e-8
        u = np.full(5, .5)
        u[active] = np.sqrt((releases[active]-low[active])/(high[active]-low[active]))
        return np.where(active, np.rint(u*(self.action_bins-1)), 0).astype(np.int64)

    def step(self, actions):
        return self._step(actions)

    def step_towards_storage(self, target_storage):
        """Track a soft END-of-day target using the policy's discrete grid.

        Each desired release uses the actually selected upstream release. The
        normal transition protects every hard bound and future deadline; an
        unreachable soft target is approached, never imposed by clipping water.
        """
        target = np.asarray(target_storage, dtype=np.float64)
        if (target.shape != (5,) or not np.isfinite(target).all() or
                (target < self.data.vmin).any() or (target > self.data.vmax).any()):
            raise ValueError('Reference target must lie within global storage bounds')
        actions = np.zeros(5, dtype=np.int64)

        def select(agent, inflow, low, high):
            desired = inflow + (self.storage[agent]-target[agent])/self.physics.dt
            fraction = np.clip((desired-low)/(high-low), 0., 1.)
            actions[agent] = int(np.rint(np.sqrt(fraction)*(self.action_bins-1)))
            return actions[agent]/(self.action_bins-1)

        return self._step(actions, action_selector=select)

    def _step(self, actions, action_selector=None):
        if self.index >= self.stop:
            raise RuntimeError("Episode finished; reset explicitly")
        actions = np.asarray(actions)
        if actions.shape not in [(5,), (5, 1)]:
            raise ValueError("Five scalar actions required")
        normalized = self.normalize_actions(actions)
        try:
            terminal_bounds = None
            bounds = {}
            if self.seasonal is not None:
                self._check_previous_storage(self.storage, self.index)
                offset = self.index-self.start
                future_lower, future_upper = self.storage_lower[offset:], self.storage_upper[offset:]
                terminal_bounds = TerminalReleaseBounds(self.data, self.physics.dt, self.storage,
                    self.data.external[self.index:self.stop], self.terminal_storage_target,
                    storage_lower=future_lower, storage_upper=future_upper)
                bounds = dict(storage_lower=future_lower[0], storage_upper=future_upper[0])
            elif self.terminal_storage_target is not None:
                terminal_bounds = TerminalReleaseBounds(self.data, self.physics.dt, self.storage,
                    self.data.external[self.index:self.stop], self.terminal_storage_target)
            result = self.physics.transition(self.storage, self.data.external[self.index], normalized,
                                             terminal_bounds=terminal_bounds,
                                             action_selector=action_selector, **bounds)
        except RuntimeError as exc:
            raise type(exc)('{}; date={} storage_m3={} actions={} target_m3={}'.format(
                exc, self.data.dates[self.index], self.storage.tolist(), actions.reshape(5).tolist(),
                None if self.terminal_storage_target is None else self.terminal_storage_target.tolist())) from exc
        if self.index+1 == self.stop and self.terminal_storage_target is not None:
            tolerance = np.maximum(1., 1e-10*np.abs(self.terminal_storage_target))
            if (np.abs(result['storage_end']-self.terminal_storage_target) > tolerance).any():
                raise AssertionError('Terminal storage target violation')
        result["date"] = self.data.dates[self.index].isoformat()
        result['normalized_actions'] = result['actions'].copy()
        # Forced actions have no policy contribution; zero is a valid placeholder index.
        result['actions'] = np.where(result['controls'] > 0, actions.reshape(5), 0).astype(np.int64)
        result["levels_end"] = self.data.levels(result["storage_end"])
        self.operation_reward.apply(result)
        infos = [dict(bad_transition=False) for _ in range(5)]
        if self.seasonal is not None:
            offset = self.index-self.start
            phases = tuple(SEASONAL_PHASES[c] for c in self._phase_codes[offset])
            period = self.data.period_labels[self.index]
            result.update(operation_period=period, seasonal_phases=phases,
                          flood_limit_active=self.seasonal.flood_limit_active[self.index].copy(),
                          refill_deadline=self.seasonal.refill_deadline[self.index].copy())
            for j, info in enumerate(infos):
                info.update(seasonal_constraints=True, operation_period=period, seasonal_phase=phases[j],
                            storage_lower_m3=float(self.storage_lower[offset,j]),
                            storage_upper_m3=float(self.storage_upper[offset,j]),
                            flood_limit_active=bool(result['flood_limit_active'][j]),
                            refill_deadline=bool(result['refill_deadline'][j]))
        self.last_transition = result
        self.storage = result["storage_end"].copy()
        self.previous_release = result["release"].copy()
        self.index += 1
        done = self.index == self.stop
        obs, state, available = self._observations()
        return obs, state, np.full((5, 1), result["reward"], dtype=np.float32), \
            np.full(5, done, dtype=bool), infos, available
