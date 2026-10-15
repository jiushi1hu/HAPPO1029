"""Interactive result windows; no figure or run-result files are exported."""
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


def make_figure(payload):
    # Tests may explicitly select Agg before calling this function.
    import matplotlib.pyplot as plt
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
        values.append([r["mean_TWh"] for r in history])
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite validation energy")
    mean = values.mean(axis=0)
    sd = values.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(mean)
    reference = {r["year"]: r["energy_TWh"] for r in payload["references"]}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.canvas.manager.set_window_title("Five-reservoir HAPPO results")
    for line in values:
        ax.plot(grid, line, color="#087f8c", alpha=.25, lw=.8)
    ax.plot(grid, mean, color="#087f8c", marker="o", ms=3, label="Validation mean")
    if len(seeds) > 1:
        ax.fill_between(grid, mean-sd, mean+sd, color="#087f8c", alpha=.15, label="Seed +/-1 SD")
    ax.axhline(np.mean([reference[y] for y in payload["validation_years"]]),
               color="#bb6c22", ls="--", label="Constant-storage reference")
    ax.set(title="2020-2024 mean annual validation generation",
           xlabel="Training episodes (one episode = one randomly sampled year)", ylabel="Generation (TWh/year)")
    ax.legend(frameon=False, fontsize=9); ax.grid(axis="y", alpha=.2)
    if not payload["tests"]:
        bx.axis("off")
        bx.text(.5,.5,"Final typical-year tests skipped\n(diagnostic run)",
                ha="center",va="center",transform=bx.transAxes,fontsize=14)
        fig.suptitle("Five-reservoir HAPPO | validation diagnostics",fontsize=13)
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
           color="#bb6c22", label="Constant-storage reference")
    bx.set_xticks(positions)
    bx.set_xticklabels([str(y)+"\n"+label.replace("_", "\n") for y,label in
                        zip(payload["test_years"], payload["test_labels"])], fontsize=9)
    bx.set(title="Final typical-year tests (user-defined categories)", ylabel="Annual generation (TWh)")
    bx.legend(frameon=False, fontsize=9); bx.grid(axis="y", alpha=.2)
    fig.suptitle("Five independent HAPPO actors | daily scale | quadratic release mapping", fontsize=13)
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
    reward_fig, ax = plt.subplots(figsize=(10,6),constrained_layout=True)
    reward_fig.canvas.manager.set_window_title('HAPPO training reward')
    # The existing reward is daily cascade MWh / reference_energy_MWh;
    # summing over the year gives exactly this energy-to-return conversion.
    draw(ax,energy*1e6/reference,'{}-episode trailing mean'.format(window),show_raw=True)
    ax.set(title='Stochastic training return | sampled training years',
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
    if args.plot_details:
        make_training_figures(payload)
    import matplotlib.pyplot as plt
    plt.show(block=True)


if __name__ == "__main__":
    main()
