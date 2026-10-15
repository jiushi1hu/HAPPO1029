"""Full-state reservoir simulation with explicit calendar bounds and reset."""
import calendar
import numpy as np
from gym.spaces import Box, Discrete
from harl.envs.cascade_reservoir.reservoir_physics import ReservoirPhysics
from harl.envs.cascade_reservoir.terminal_storage import TerminalReleaseBounds


class CascadeReservoirEnv:
    n_agents = 5

    def __init__(self, data, config, end_year=None, start_year=None, action_bins=None):
        self.data = data
        self.physics = ReservoirPhysics(data, config)
        self.terminal_storage_mode = config.get('terminal_storage_mode', 'free')
        if self.terminal_storage_mode not in ('free', 'initial'):
            raise ValueError('Terminal storage mode must be free or initial')
        start_year = data.start_year if start_year is None else int(start_year)
        end_year = start_year if end_year is None else int(end_year)
        if self.terminal_storage_mode == 'initial' and start_year != end_year:
            raise ValueError('Return-to-initial storage requires a single-year episode')
        self.start, self.stop = data.indices(start_year, end_year)
        # A finite-horizon policy needs time-to-go, not just day of year.
        # Keep the scale fixed across training, validation and test horizons.
        self.time_scale_days = 366  # Fixed annual scale, including leap days.
        self.observation_space = [Box(-np.inf, np.inf, (18,), dtype=np.float32) for _ in range(5)]
        self.share_observation_space = self.observation_space
        if action_bins is not None and (isinstance(action_bins, bool) or
                not isinstance(action_bins, (int, np.integer)) or action_bins < 2):
            raise ValueError('action_bins must be an integer >= 2')
        self.action_bins = action_bins
        self.action_space = [Discrete(action_bins) if action_bins is not None else
                            Box(0., 1., (1,), dtype=np.float32) for _ in range(5)]
        self.reset()

    def reset(self):
        self.index = self.start
        self.storage = self.data.initial_storage(self.data.dates[self.start].year)
        self.terminal_storage_target = self.storage.copy() if self.terminal_storage_mode == 'initial' else None
        self.previous_release = np.zeros(5)
        self.last_transition = None
        return self._observations()

    def observation(self):
        if self.index == self.stop:
            return np.zeros(18, dtype=np.float32)
        day = self.data.dates[self.index]
        phase = 2. * np.pi * (day.timetuple().tm_yday - 1) / (366 if calendar.isleap(day.year) else 365)
        return np.concatenate([
            (self.storage - self.data.vmin) / (self.data.vmax - self.data.vmin),
            (self.data.external[self.index] - self.data.flow_mean) / self.data.flow_std,
            self.previous_release / self.data.release_cap,
            [np.sin(phase), np.cos(phase)],
            [(self.stop - self.index) / self.time_scale_days],
        ]).astype(np.float32)

    def _observations(self):
        obs = np.repeat(self.observation()[None, :], 5, axis=0)
        return obs, obs.copy(), None

    def normalize_actions(self, actions):
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape not in ((5,), (5,1)) or not np.isfinite(actions).all():
            raise ValueError('Five finite scalar actions required')
        actions = actions.reshape(5)
        if self.action_bins is None:
            if ((actions < 0.) | (actions > 1.)).any():
                raise ValueError('Normalized actions must lie in [0, 1]')
            return actions
        if ((actions != np.floor(actions)) | (actions < 0) | (actions >= self.action_bins)).any():
            raise ValueError('Discrete actions must be integer indices in [0, action_bins-1]')
        return actions / (self.action_bins-1)

    def actions_from_releases(self, releases, low, high):
        """Invert given feasible bounds; round to nearest normalized grid index.

        Off-grid releases are quantized in u, not in m3/s. A collapsed interval
        has canonical index zero (or control .5 for the continuous interface).
        """
        releases, low, high = [np.asarray(x, dtype=np.float64) for x in (releases, low, high)]
        if any(x.shape != (5,) or not np.isfinite(x).all() for x in (releases, low, high)):
            raise ValueError('Expected finite five-vectors for release bounds')
        if ((high < low) | (releases < low) | (releases > high)).any():
            raise ValueError('Release outside feasible bounds')
        active = high-low > 1e-8
        u = np.full(5, .5)
        u[active] = np.sqrt((releases[active]-low[active])/(high[active]-low[active]))
        if self.action_bins is None:
            return u
        return np.where(active, np.rint(u*(self.action_bins-1)), 0).astype(np.int64)

    def step(self, actions):
        if self.index >= self.stop:
            raise RuntimeError("Episode finished; reset explicitly")
        actions = np.asarray(actions)
        if actions.shape not in [(5,), (5, 1)]:
            raise ValueError("Five scalar actions required")
        normalized = self.normalize_actions(actions)
        try:
            terminal_bounds = None
            if self.terminal_storage_target is not None:
                terminal_bounds = TerminalReleaseBounds(self.data, self.physics.dt, self.storage,
                    self.data.external[self.index:self.stop], self.terminal_storage_target)
            result = self.physics.transition(self.storage, self.data.external[self.index], normalized,
                                             terminal_bounds=terminal_bounds)
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
        if self.action_bins is not None:
            # Forced actions have no policy contribution; zero is a valid placeholder index.
            result['actions'] = np.where(result['controls'] > 0, actions.reshape(5), 0).astype(np.int64)
        result["levels_end"] = self.data.levels(result["storage_end"])
        self.last_transition = result
        self.storage = result["storage_end"].copy()
        self.previous_release = result["release"].copy()
        self.index += 1
        done = self.index == self.stop
        obs, state, available = self._observations()
        return obs, state, np.full((5, 1), result["reward"], dtype=np.float32), \
            np.full(5, done, dtype=bool), [dict(bad_transition=False) for _ in range(5)], available
