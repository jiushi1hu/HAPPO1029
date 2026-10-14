"""One interactive window; no figure or run-result files are exported."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-backend", action="store_true")
    args = parser.parse_args()
    backend = select_interactive_backend()
    if args.check_backend:
        print("Interactive plotting backend available: " + backend, flush=True)
        return
    payload = json.load(sys.stdin)
    make_figure(payload)
    import matplotlib.pyplot as plt
    plt.show(block=True)


if __name__ == "__main__":
    main()
