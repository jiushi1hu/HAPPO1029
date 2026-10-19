"""Interactive result windows; figures are not exported."""
import argparse
import importlib
import json
import sys
import numpy as np
import matplotlib


def select_interactive_backend():
    backend = "MacOSX" if sys.platform == "darwin" else "TkAgg"
    matplotlib.use(backend, force=True)
    importlib.import_module("matplotlib.backends.backend_" + backend.lower())
    return backend


def comparison_data(payload):
    """Reject mixed operating modes rather than plotting an unfair comparison."""
    seasonal = payload.get('seasonal_constraints', False)
    if not isinstance(seasonal, bool):
        raise ValueError('seasonal_constraints must be a boolean')
    kind = 'seasonal_target_tracking' if seasonal else 'constant_storage'
    if payload.get('reference_type', kind) != kind:
        raise ValueError('Reference type does not match operating mode')
    label = payload.get('reference_label', 'Seasonal rule reference' if seasonal else 'Constant-storage reference')
    penalties = payload.get('reward_penalties',{'enabled':False})
    reference = {}
    rows = list(payload['references']) + list(payload['tests'])
    for history in payload['histories']:
        rows.extend(row for checkpoint in history for row in checkpoint.get('years', []))
    for row in rows:
        row_penalties = row.get('reward_penalties',{'enabled':False})
        if (row_penalties.get('enabled',False) != penalties.get('enabled',False) or
                penalties.get('enabled',False) and row_penalties != penalties):
            raise ValueError('Policy and reference must use the same reward penalties')
        if row.get('seasonal_constraints', False) != seasonal:
            raise ValueError('Policy and reference must use the same seasonal constraints')
        if seasonal and row.get('action_bins') != payload.get('action_bins'):
            raise ValueError('Policy and reference must use the same action grid')
        for key in ('head_model', 'head_data_status'):
            if key in payload and row.get(key) != payload[key]:
                raise ValueError('Policy and reference must use the same '+key)
        if seasonal and row.get('terminal_storage_mode') != payload.get('terminal_storage_mode'):
            raise ValueError('Policy and reference must use the same terminal storage mode')
        if row.get('feasible', True) is not True or row.get('violation_count', 0) != 0:
            raise ValueError('Cannot compare infeasible evaluation trajectories')
    for row in payload['references']:
        year = row['year']
        if year in reference or not np.isfinite(row['energy_TWh']):
            raise ValueError('Reference years must be unique with finite energy')
        if row.get('reference_type', kind) != kind:
            raise ValueError('Mixed reference types')
        reference[year] = row['energy_TWh']
    required = list(payload['validation_years']) + (list(payload['test_years']) if payload['tests'] else [])
    if not required or any(year not in reference for year in required):
        raise ValueError('Reference missing for an evaluated year')
    return reference, label, 'seasonal constraints' if seasonal else 'global storage bounds'


def validation_series(payload, metric):
    """Align validation checkpoints by episode, including the untrained point."""
    seeds = payload["seeds"]
    histories = payload["histories"]
    if len(seeds) != len(histories) or not seeds:
        raise ValueError("One nonempty validation history per seed required")
    grid = np.array([r["episode"] for r in histories[0]])
    if not len(grid) or np.any(np.diff(grid) <= 0):
        raise ValueError("Validation episode checkpoints must be increasing")
    values = []
    for history in histories:
        if not np.array_equal(grid, [r["episode"] for r in history]):
            raise ValueError("Seeds must have matching episode checkpoints")
        if any(metric not in r for r in history):
            raise ValueError('Validation plot requires recorded '+metric+' values')
        values.append([r[metric] for r in history])
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite validation metric: '+metric)
    return seeds, grid, values


def make_figure(payload):
    # Tests may explicitly select Agg before calling this function.
    import matplotlib.pyplot as plt
    seeds, grid, values = validation_series(payload, 'mean_TWh')
    mean = values.mean(axis=0)
    sd = values.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(mean)
    reference, reference_label, operating_mode = comparison_data(payload)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.canvas.manager.set_window_title("Five-reservoir HAPPO results")
    for line in values:
        ax.plot(grid, line, color="#087f8c", alpha=.25, lw=.8)
    ax.plot(grid, mean, color="#087f8c", marker="o", ms=3, label="Validation mean")
    if len(seeds) > 1:
        ax.fill_between(grid, mean-sd, mean+sd, color="#087f8c", alpha=.15, label="Seed +/-1 SD")
    years = payload['validation_years']
    year_label = '{}-{}'.format(min(years), max(years))
    ax.set(title=year_label+" mean annual validation generation",
           xlabel="Training episodes (one episode = one randomly sampled year)", ylabel="Generation (TWh/year)")
    ax.legend(frameon=False, fontsize=9); ax.grid(axis="y", alpha=.2)
    if payload.get('reward_penalties',{}).get('enabled',False):
        operating_mode += ' | selected by penalized return'
    if not payload["tests"]:
        bx.axis("off")
        bx.text(.5,.5,"Final typical-year tests skipped\n(diagnostic run)",
                ha="center",va="center",transform=bx.transAxes,fontsize=14)
        fig.suptitle("Five-reservoir HAPPO | "+operating_mode+" | validation diagnostics",fontsize=13)
        return fig
    test_values = []
    for year, label in zip(payload["test_years"], payload["test_labels"]):
        rows = [r for r in payload["tests"] if r["year"] == year and r["typical_label"] == label]
        if len(rows) != len(seeds) or {r["seed"] for r in rows} != set(seeds):
            raise ValueError("One typical-year result per seed required")
        test_values.append([r["energy_TWh"] for r in rows])
    test_values = np.array(test_values, dtype=float)
    if not np.isfinite(test_values).all():
        raise ValueError("Non-finite test energy")
    test_sd = test_values.std(axis=1, ddof=1) if len(seeds) > 1 else np.zeros(len(test_values))
    positions = np.arange(len(test_values))
    bx.bar(positions-.18, test_values.mean(axis=1), .36, yerr=test_sd,
           capsize=3, color="#087f8c", label="Selected HAPPO (seed SD)")
    bx.bar(positions+.18, [reference[y] for y in payload["test_years"]], .36,
           color="#bb6c22", label=reference_label)
    bx.set_xticks(positions)
    bx.set_xticklabels([str(y)+"\n"+label.replace("_", "\n") for y,label in
                        zip(payload["test_years"], payload["test_labels"])], fontsize=9)
    bx.set(title="Final typical-year tests (user-defined categories)", ylabel="Annual generation (TWh)")
    bx.legend(frameon=False, fontsize=9); bx.grid(axis="y", alpha=.2)
    fig.suptitle("Five independent HAPPO actors | daily scale | "+operating_mode, fontsize=13)
    return fig


def make_validation_reward_figure(payload):
    """Plot recorded, undiscounted validation returns used for model selection."""
    import matplotlib.pyplot as plt
    seeds, grid, values = validation_series(payload, 'mean_reward')
    mean = values.mean(axis=0)
    fig, ax = plt.subplots(figsize=(10,6), constrained_layout=True)
    fig.canvas.manager.set_window_title('HAPPO validation net reward')
    for seed, line in zip(seeds, values):
        ax.plot(grid, line, alpha=.4, lw=.9, label='Seed {}'.format(seed))
    ax.plot(grid, mean, color='#087f8c', lw=2, marker='o', ms=3, label='Validation mean')
    if len(seeds) > 1:
        sd = values.std(axis=0, ddof=1)
        ax.fill_between(grid, mean-sd, mean+sd, color='#087f8c', alpha=.15,
                        label='Seed +/-1 SD')
    years = payload['validation_years']
    ax.set(title='{}-{} mean annual validation net reward'.format(min(years),max(years)),
           xlabel='Training episodes (one episode = one randomly sampled year)',
           ylabel='Undiscounted annual net reward (reward units)')
    ax.grid(alpha=.2)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    return fig


def make_training_figures(payload, window=20):
    """Extra windows from existing in-memory training metrics; no model changes."""
    import matplotlib.pyplot as plt
    histories = payload['training_histories']
    if not histories or len(histories) != len(payload['seeds']):
        raise ValueError('One nonempty training history per seed required')
    grid = np.array([r['iteration'] for r in histories[0]])
    if not len(grid) or not np.array_equal(grid, np.arange(1,len(grid)+1)):
        raise ValueError('Complete training episodes starting at one required')
    for rows in histories:
        if not np.array_equal(grid,[r['iteration'] for r in rows]):
            raise ValueError('Seeds must have matching training episodes')
    reference = float(payload['reference_energy_MWh'])
    if not np.isfinite(reference) or reference <= 0 or window < 1:
        raise ValueError('Positive reward reference and smoothing window required')

    def values(key):
        result = np.asarray([[r[key] for r in rows] for rows in histories],dtype=float)
        if not np.isfinite(result).all():
            raise ValueError('Non-finite training metric: '+key)
        return result

    def smooth(series):
        cumulative = np.pad(np.cumsum(series,axis=1),((0,0),(1,0)))
        stop = np.arange(1,len(grid)+1)
        start = np.maximum(0,stop-window)
        return (cumulative[:,stop]-cumulative[:,start])/(stop-start)

    def draw(ax, series, label, show_raw=False):
        if show_raw:
            ax.plot(grid,series.mean(axis=0),color='#087f8c',alpha=.2,lw=.7,
                    label='Raw episode mean across seeds')
        rolled = smooth(series)
        for line in rolled:
            ax.plot(grid,line,color='#087f8c',alpha=.25,lw=.8)
        mean = rolled.mean(axis=0)
        ax.plot(grid,mean,color='#087f8c',label=label)
        if len(histories)>1:
            sd = rolled.std(axis=0,ddof=1)
            ax.fill_between(grid,mean-sd,mean+sd,color='#087f8c',alpha=.15,
                            label='Smoothed seed +/-1 SD')
        ax.set_xlabel('Training episodes (randomly sampled years)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)

    energy = values('episode_energy_TWh')
    if all('episode_reward' in row for rows in histories for row in rows):
        returns = values('episode_reward')
    elif payload.get('reward_penalties',{}).get('enabled',False):
        raise ValueError('Penalized reward plots require actual episode_reward values')
    else:
        # Backward compatibility for generation-only historical payloads.
        returns = energy*1e6/reference
    reward_fig, ax = plt.subplots(figsize=(10,6),constrained_layout=True)
    reward_fig.canvas.manager.set_window_title('HAPPO training reward')
    draw(ax,returns,'{}-episode trailing mean'.format(window),show_raw=True)
    objective = 'penalized return' if payload.get('reward_penalties',{}).get('enabled',False) else 'return'
    ax.set(title='Stochastic training '+objective+' | sampled training years',
           ylabel='Episode return (sum of daily rewards)')

    energy_fig, ax = plt.subplots(figsize=(10,6),constrained_layout=True)
    energy_fig.canvas.manager.set_window_title('HAPPO training generation')
    draw(ax,energy,'{}-episode trailing mean'.format(window),show_raw=True)
    ax.set(title='Stochastic training generation | sampled training years',
           ylabel='Annual cascade generation (TWh)')

    optimization_fig, axes = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    optimization_fig.canvas.manager.set_window_title('HAPPO optimization diagnostics')
    for i in range(5):
        for ax, suffix in [(axes[0,0],'policy_loss'),(axes[1,0],'dist_entropy'),
                           (axes[1,1],'actor_grad_norm')]:
            ax.plot(grid,smooth(values('actor{}_{}'.format(i,suffix))).mean(axis=0),
                    label='Actor {}'.format(i+1))
    axes[0,1].plot(grid,smooth(values('critic_value_loss')).mean(axis=0),label='Critic')
    for ax,title,ylabel in [(axes[0,0],'Actor loss','Policy loss'),
            (axes[0,1],'Critic loss','Value loss'),(axes[1,0],'Policy entropy','Entropy (nats)'),
            (axes[1,1],'Actor gradients before clipping','Gradient norm')]:
        ax.set(title=title,xlabel='Training episodes',ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    optimization_fig.suptitle('{}-episode trailing means, averaged across seeds'.format(window))
    return [reward_fig,energy_fig,optimization_fig]


def make_reward_scale_figures(payload, window=20):
    """Plot applied reward components and ratios without rerunning any policy."""
    import matplotlib.pyplot as plt
    histories = payload['episode_histories']
    if (not histories or len(histories) != len(payload['seeds']) or
            isinstance(window, bool) or not isinstance(window, int) or window < 1):
        raise ValueError('One episode history per seed and a positive integer window required')
    grid = np.array([r['episode'] for r in histories[0]])
    if not len(grid) or not np.array_equal(grid, np.arange(1, len(grid)+1)):
        raise ValueError('Complete episode metrics starting at one required')
    for seed, history in zip(payload['seeds'], histories):
        if not np.array_equal(grid, [r['episode'] for r in history]):
            raise ValueError('Seeds must have matching episode metrics')
        if any(r['seed'] != seed for r in history):
            raise ValueError('Episode metrics must match their seed')

    def values(key):
        result = np.asarray([[r[key] for r in rows] for rows in histories], dtype=float)
        if not np.isfinite(result).all():
            raise ValueError('Non-finite episode metric: '+key)
        return result

    power, total = values('power_reward'), values('total_reward')
    costs = [values(k+'_penalty') for k in ('band', 'rise', 'fall')]
    if (power < 0.).any() or any((cost < 0.).any() for cost in costs):
        raise ValueError('Power Reward and penalty costs must be nonnegative')
    if not np.allclose(total, power-sum(costs), rtol=1e-9, atol=1e-9):
        raise ValueError('Reward components do not sum to Total Reward')
    stop = np.arange(1, len(grid)+1)
    start = np.maximum(0, stop-window)

    def rolling_sum(series):
        cumulative = np.pad(np.cumsum(series, axis=1), ((0,0),(1,0)))
        return cumulative[:,stop]-cumulative[:,start]

    def average(series):
        return rolling_sum(series)/(stop-start)

    def seed_mean(series):
        return np.ma.masked_invalid(series).mean(axis=0).filled(np.nan)

    def draw(ax, raw, rolled):
        ax.plot(grid, seed_mean(raw), color='#087f8c', alpha=.22, lw=.7,
                label='Raw episode mean')
        for line in rolled:
            ax.plot(grid, line, color='#087f8c', alpha=.25, lw=.8)
        mean = seed_mean(rolled)
        ax.plot(grid, mean, color='#087f8c', lw=1.8, marker='o', ms=2.5,
                label='{}-episode window'.format(window))
        if len(histories) > 1:
            sd = np.ma.masked_invalid(rolled).std(axis=0, ddof=1).filled(np.nan)
            ax.fill_between(grid, mean-sd, mean+sd, color='#087f8c', alpha=.15,
                            label='Seed +/-1 SD')
        ax.set_xlabel('Training episodes (sampled years)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)

    components_fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    components_fig.canvas.manager.set_window_title('HAPPO reward components')
    for ax, name, series in zip(axes.flat, ['Power Reward', 'Band penalty', 'Rise penalty',
                                          'Fall penalty', 'Total Reward'], [power]+costs+[total]):
        draw(ax, series, average(series))
        ax.set(title=name, ylabel='Undiscounted episode reward units')
    axes.flat[-1].axis('off')
    axes.flat[-1].text(.05,.6,'Total = Power - Band - Rise - Fall\n\nPenalties are positive applied costs.\nFailed episodes contain executed days only.',
                        transform=axes.flat[-1].transAxes, va='center', fontsize=10)
    components_fig.suptitle('Training reward components | episode sums')

    ratios_fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    ratios_fig.canvas.manager.set_window_title('HAPPO penalties relative to Power Reward')
    def percent(numerator, denominator):
        return np.divide(100.*numerator, denominator, out=np.full_like(denominator, np.nan),
                         where=denominator > 0.)
    for ax, name, cost in zip(axes.flat, ['Band', 'Rise', 'Fall', 'All soft penalties'], costs+[sum(costs)]):
        # Ratios of window sums preserve energy weighting. Averaging episode
        # percentages would over-weight low-generation years.
        draw(ax, percent(cost, power), percent(rolling_sum(cost), rolling_sum(power)))
        ax.set(title=name+' / Power Reward', ylabel='Applied penalty / Power Reward (%)')
    ratios_fig.suptitle('Penalty scaling | window cost / window Power Reward\nZero denominators are gaps')
    return [components_fig, ratios_fig]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-backend", action="store_true")
    parser.add_argument('--plot-details', action='store_true')
    args = parser.parse_args()
    backend = select_interactive_backend()
    if args.check_backend:
        print("Interactive plotting backend available: " + backend, flush=True)
        return
    payload = json.load(sys.stdin)
    make_figure(payload)
    if payload.get('reward_penalties',{}).get('enabled',False):
        make_validation_reward_figure(payload)
    if 'episode_histories' in payload:
        make_reward_scale_figures(payload)
    if args.plot_details:
        make_training_figures(payload)
    import matplotlib.pyplot as plt
    plt.show(block=True)


if __name__ == "__main__":
    main()
