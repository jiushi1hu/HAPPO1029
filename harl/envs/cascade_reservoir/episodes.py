"""Explicit fixed-day episodes without annual resets or split leakage."""
import copy
import numpy as np


def positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(name+' must be a positive integer')
    return int(value)


def training_windows(data, days):
    """Jan-1 starts whose entire consecutive daily window belongs to training."""
    days = positive_integer(days, 'episode_days')
    allowed = set(data.training_years)
    windows = {}
    for year in sorted(allowed):
        start, _ = data.indices(year, year)
        stop = start+days
        if stop > len(data.dates):
            continue
        years = list(dict.fromkeys(d.year for d in data.dates[start:stop]))
        if set(years).issubset(allowed):
            windows[int(year)] = years
    if not windows:
        raise ValueError('No complete training-only windows for episode_days={}'.format(days))
    return windows


class EpisodeData:
    """Read-only daily view; static physics and initial-storage inputs are shared.

    Ordered years can be nonconsecutive for the representative held-out test
    sequence. Their source dates and operating rules are retained explicitly.
    No hydrology is interpolated and no leap days are deleted.
    """

    def __init__(self, source, years, days):
        days = positive_integer(days, 'episode_days')
        if not years or len(set(years)) != len(years):
            raise ValueError('Episode source years must be nonempty and unique')
        indices = np.concatenate([np.arange(*source.indices(year, year)) for year in years])
        if len(indices) < days:
            raise ValueError('Episode source years do not contain the requested daily horizon')
        self.source = source
        # Keep each source year's full operating calendar; the environment
        # limits execution and reachability to exactly `days` transitions.
        self.source_indices = indices
        self.dates = tuple(source.dates[i] for i in self.source_indices)
        self.years = np.array([d.year for d in self.dates])
        self.source_years = list(dict.fromkeys(self.years.tolist()))
        self.start_year, self.end_year = min(self.source_years), max(self.source_years)
        self.external = source.external[self.source_indices].copy()
        self.external.setflags(write=False)
        self.period_labels = tuple(source.period_labels[i] for i in self.source_indices)
        self.seasonal_operation = None

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, 'source'), name)

    def indices(self, start_year, end_year):
        selected = np.flatnonzero((self.years >= start_year) & (self.years <= end_year))
        if not len(selected):
            raise ValueError('Year selection is outside this episode')
        if not np.array_equal(selected, np.arange(selected[0], selected[-1]+1)):
            raise ValueError('Year selection is not contiguous in this episode sequence')
        return int(selected[0]), int(selected[-1])+1

    def load_seasonal_operation(self):
        if self.seasonal_operation is None:
            original = self.source.load_seasonal_operation()
            view = copy.copy(original)
            view.data = self
            for name in ('storage_lower', 'storage_upper', 'flood_limit_active',
                         'refill_deadline', 'reference_subphases'):
                value = getattr(original, name)[self.source_indices].copy()
                value.setflags(write=False)
                setattr(view, name, value)
            view.reference = {k:v[self.source_indices].copy() for k,v in original.reference.items()}
            for value in view.reference.values():
                value.setflags(write=False)
            self.seasonal_operation = view
        return self.seasonal_operation


def episode_environment(data, physical, years, days, action_bins):
    from harl.envs.cascade_reservoir.cascade_reservoir_env import CascadeReservoirEnv
    view = EpisodeData(data, years, days)
    return CascadeReservoirEnv(view, physical, start_year=years[0], end_year=years[0],
                               action_bins=action_bins, episode_days=days)
