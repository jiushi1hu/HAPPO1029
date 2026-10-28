"""Unsmoothed multi-seed training curves for complete cascade episodes."""
import json
import sys
import numpy as np


def make_figure(payload):
    import matplotlib.pyplot as plt
    histories = payload['histories']
    if (not histories or len(histories) != len(payload['seeds']) or
            any(not rows for rows in histories)):
        raise ValueError('One nonempty parallel history per seed required')
    grid = np.array([r['training_steps'] for r in histories[0]], dtype=float)
    if not np.isfinite(grid).all() or np.any(grid <= 0) or np.any(np.diff(grid) <= 0):
        raise ValueError('Parallel checkpoints must increase; training curves start after collection')
    if any(not np.array_equal(grid, [r['training_steps'] for r in rows]) for rows in histories):
        raise ValueError('Parallel curves require matched training-step grids')
    gamma = float(payload['gamma'])
    if not np.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError('Parallel curves require gamma in [0, 1]')
    metrics = ('episode_reward', 'episode_energy_TWh')
    values = [np.asarray([[r[metric] for r in rows] for rows in histories], dtype=float)
              for metric in metrics]
    if not all(np.isfinite(series).all() for series in values):
        raise ValueError('Non-finite parallel curve metric')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    prefix = 'Mean collected episode'
    for ax, series, label in zip(axes, values,
            (prefix+' net return (gamma={:g})'.format(gamma), prefix+' generation (TWh)')):
        for seed, row in zip(payload['seeds'], series):
            ax.plot(grid, row, alpha=.35, lw=.8, label='Seed {}'.format(seed))
        mean = series.mean(axis=0)
        ax.plot(grid, mean, lw=2, label='Seed mean')
        if len(series) > 1:
            sd = series.std(axis=0, ddof=1)
            ax.fill_between(grid, mean-sd, mean+sd, alpha=.15, label='Seed +/-1 SD')
        days = payload['episode_horizon_days']
        horizon = 'Five-year (1,825-day)' if days == 1825 else '{}-day diagnostic'.format(days)
        ax.set(xlabel='Collected daily environment transitions', ylabel=label,
               title=horizon+' stochastic training episodes (in-sample)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    return fig


if __name__ == '__main__':
    make_figure(json.load(sys.stdin))
    import matplotlib.pyplot as plt
    plt.show()
