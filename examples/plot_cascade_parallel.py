"""Unsmoothed multi-seed training curves for complete cascade episodes."""
import argparse
import json
from pathlib import Path
import re
import sys
import numpy as np


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(name+' must be a positive integer')
    return int(value)


def load_episode_log(path):
    """Stream update summaries; daily trajectories are never retained in memory."""
    configuration, histories, updates = None, {}, {}
    # Step writers put record_type first. Other key orders still use the JSON
    # parser below, so existing logs do not depend on object-key order.
    step_prefix = re.compile(r'^\s*\{\s*"record_type"\s*:\s*"(?:training_step|evaluation_step)"\s*[,}]')
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            if step_prefix.match(line):
                if not line.endswith('\n'):
                    print('Ignoring an incomplete final log record while reading a running experiment.',
                          file=sys.stderr)
                    break
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                if not line.endswith('\n'):
                    print('Ignoring an incomplete final log record while reading a running experiment.',
                          file=sys.stderr)
                    break
                raise ValueError('Invalid JSON at line {}'.format(line_number)) from exc
            if not isinstance(row, dict):
                raise ValueError('Log records must be JSON objects at line {}'.format(line_number))
            kind = row.get('record_type')
            if kind not in ('run_configuration', 'training_update'):
                continue
            try:
                if kind == 'run_configuration':
                    if configuration is not None:
                        raise ValueError('A plot input must contain exactly one run configuration')
                    gamma = float(row['config']['algo']['gamma'])
                    if not np.isfinite(gamma) or not 0 <= gamma <= 1:
                        raise ValueError('Gamma must be finite and in [0, 1]')
                    seeds = row['config']['experiment']['seeds']
                    if (not isinstance(seeds, list) or not seeds or
                            any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds) or
                            len(set(seeds)) != len(seeds)):
                        raise ValueError('Configured seeds must be unique integers')
                    configuration = dict(gamma=gamma, configured_seeds=seeds,
                        episode_horizon_days=positive_integer(row['episode_horizon_days'], 'episode_horizon_days'),
                        parallel_envs=positive_integer(row['parallel_environments'], 'parallel_environments'))
                    continue
                if configuration is None:
                    raise ValueError('A run_configuration must precede training updates')
                seed = row['seed']
                if isinstance(seed, bool) or not isinstance(seed, int) or seed not in configuration['configured_seeds']:
                    raise ValueError('Training update seed is absent from the run configuration')
                if (row['episode_horizon_days'] != configuration['episode_horizon_days'] or
                        row['parallel_envs'] != configuration['parallel_envs']):
                    raise ValueError('Training update horizon/workers disagree with the run configuration')
                step = positive_integer(row['training_steps'], 'training_steps')
                update = positive_integer(row['update'], 'update')
                history = histories.setdefault(seed, [])
                if history and (step <= history[-1]['training_steps'] or update <= updates[seed]):
                    raise ValueError('Duplicate or non-increasing training update for seed {}'.format(seed))
                reward = float(row['episode_reward_mean'])
                generation = float(row['episode_generation_TWh_mean'])
                if not np.isfinite([reward, generation]).all():
                    raise ValueError('Non-finite training curve metric')
                history.append(dict(training_steps=step, episode_reward=reward,
                                    episode_energy_TWh=generation))
                updates[seed] = update
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError('Invalid plotting input at line {}: {}'.format(line_number, exc)) from exc
    if configuration is None:
        raise ValueError('No run_configuration found in the episode log')
    seeds = [seed for seed in configuration['configured_seeds'] if histories.get(seed)]
    if not seeds:
        raise ValueError('No completed training_update records yet; plot after the first HAPPO update')
    return dict(configuration, seeds=seeds, histories=[histories[seed] for seed in seeds])


def make_figure(payload):
    import matplotlib.pyplot as plt
    histories = payload['histories']
    if (not histories or len(histories) != len(payload['seeds']) or
            len(set(payload['seeds'])) != len(payload['seeds']) or any(not rows for rows in histories)):
        raise ValueError('One nonempty parallel history per seed required')
    grids = [np.array([r['training_steps'] for r in rows], dtype=float) for rows in histories]
    if any(not np.isfinite(grid).all() or np.any(grid <= 0) or np.any(np.diff(grid) <= 0)
           for grid in grids):
        raise ValueError('Training steps must increase; curves start after collection')
    # A running multi-seed experiment can have completed only some seeds or
    # updates. Show every observed point and average only common observed steps.
    common_steps = set(grids[0])
    for grid in grids[1:]:
        common_steps.intersection_update(grid)
    mean_grid = np.array(sorted(common_steps), dtype=float)
    gamma = float(payload['gamma'])
    if not np.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError('Parallel curves require gamma in [0, 1]')
    days = positive_integer(payload['episode_horizon_days'], 'episode_horizon_days')
    metrics = ('episode_reward', 'episode_energy_TWh')
    values = [[np.asarray([r[metric] for r in rows], dtype=float) for rows in histories]
              for metric in metrics]
    if not all(np.isfinite(row).all() for series in values for row in series):
        raise ValueError('Non-finite parallel curve metric')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    prefix = 'Mean collected episode'
    for ax, series, label in zip(axes, values,
            (prefix+' net return (gamma={:g})'.format(gamma), prefix+' generation (TWh)')):
        for seed, grid, row in zip(payload['seeds'], grids, series):
            ax.plot(grid, row, alpha=.35, lw=.8, marker='o' if len(grid) == 1 else None,
                    label='Seed {}'.format(seed))
        if len(mean_grid):
            aligned = np.array([row[np.searchsorted(grid, mean_grid)] for grid, row in zip(grids, series)])
            mean = aligned.mean(axis=0)
            configured = payload.get('configured_seeds', payload['seeds'])
            mean_label = ('Seed mean' if len(configured) == len(series) else
                          'Available-seed mean (n={})'.format(len(series)))
            ax.plot(mean_grid, mean, lw=2, marker='o' if len(mean_grid) == 1 else None, label=mean_label)
            if len(series) > 1:
                sd = aligned.std(axis=0, ddof=1)
                ax.fill_between(mean_grid, mean-sd, mean+sd, alpha=.15, label='Seed +/-1 SD')
        horizon = 'Five-year (1,825-day)' if days == 1825 else '{}-day diagnostic'.format(days)
        ax.set(xlabel='Collected daily environment transitions', ylabel=label,
               title=horizon+' stochastic training episodes (in-sample)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode-log', type=Path, help='Read saved JSONL training-update summaries')
    parser.add_argument('--output', type=Path, help='Save the figure, for example runs/training_curves.png')
    parser.add_argument('--no-show', action='store_true', help='Save without opening an interactive window')
    args = parser.parse_args()
    if args.no_show and args.output is None:
        parser.error('--no-show requires --output')
    if args.no_show:
        import matplotlib
        matplotlib.use('Agg', force=True)
    try:
        payload = load_episode_log(args.episode_log) if args.episode_log else json.load(sys.stdin)
        figure = make_figure(payload)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(str(args.output), dpi=180)
            print('Saved training curves to '+str(args.output.resolve()))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    import matplotlib.pyplot as plt
    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


if __name__ == '__main__':
    main()
