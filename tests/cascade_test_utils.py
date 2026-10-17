"""Explicit legacy fixtures; seasonal/default behavior has separate regressions."""
from examples.train_cascade import load_configuration as load_default_configuration


def load_configuration():
    config, physical = load_default_configuration()
    physical['seasonal_constraints'] = False
    physical['reward_penalties']['enabled'] = False
    return config, physical
