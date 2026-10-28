"""
Make figures from the CSVs written by accuracy.py and performance.py. Does not
run tsdate, so figures can be restyled without re-running anything.

Example:
  python evaluation/plot.py            # plots everything found in results/
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import common
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

LEVELS = (0.5, 0.8, 0.9, 0.95)
MODE_LABELS = {"true": "true topology", "inferred": "tsinferred topology"}


def label(mode):
    return MODE_LABELS.get(mode, mode)


def param_label(sweep, value):
    return f"n={value:g}" if sweep == "sample_size" else f"{sweep}={value:g}"


def title_of(df, path):
    quick = " (quick run, not representative)" if path.stem.endswith("_quick") else ""
    return f"tsdate method: {method_of(df)}{quick}"


def method_of(df):
    return df["method"].iloc[0] if "method" in df else "variational_gamma"


def gamma_cdf(df):
    """CDF of each node's moment-matched gamma posterior at the true time."""
    shape = df.post_mean**2 / df.post_var
    scale = df.post_var / df.post_mean
    return scipy.stats.gamma.cdf(df.true_time, a=shape, scale=scale)


def metrics(df):
    ok = (df.true_time > 0) & (df.est_time > 0)
    df = df[ok]
    logerr = np.log(df.est_time) - np.log(df.true_time)
    pit = gamma_cdf(df)
    out = {
        "n_nodes": len(df),
        "rmse_log": np.sqrt(np.mean(logerr**2)),
        "bias_log": logerr.mean(),
        "corr_log": np.corrcoef(np.log(df.est_time), np.log(df.true_time))[0, 1],
    }
    for lev in LEVELS:
        lo, hi = (1 - lev) / 2, 1 - (1 - lev) / 2
        out[f"cov_{lev}"] = np.mean((pit > lo) & (pit < hi))
    return pd.Series(out)


AGE_EDGES = [0, 10, 100, 1000, 10_000, np.inf]
AGE_LABELS = ["<10", "10-100", "100-1k", "1k-10k", ">10k"]


def plot_age_bands(df, path, modes, values, xlabel, log_x):
    """Error, bias and node share per true-age band, one row per topology."""
    df = df[(df.true_time > 0) & (df.est_time > 0)].copy()
    df["band"] = pd.cut(df.true_time, AGE_EDGES, labels=AGE_LABELS, right=False)
    df["logerr"] = np.log(df.est_time) - np.log(df.true_time)
    colours = plt.cm.viridis(np.linspace(0, 0.9, len(AGE_LABELS)))
    fig, axes = plt.subplots(
        len(modes), 3, figsize=(13, 3.4 * len(modes)), squeeze=False, sharex=True
    )
    for r, mode in enumerate(modes):
        d = df[df["mode"] == mode]
        per_seed = d.groupby(["value", "seed", "band"], observed=False).logerr
        rmse = per_seed.apply(lambda x: np.sqrt(np.mean(x**2)) if len(x) else np.nan)
        bias = per_seed.mean()
        count = per_seed.size().unstack("band")
        share = count.div(count.sum(axis=1), axis=0).groupby("value").mean()
        stats = [
            rmse.groupby(["value", "band"], observed=False).mean().unstack("band"),
            bias.groupby(["value", "band"], observed=False).mean().unstack("band"),
            share,
        ]
        titles = ["RMSE (log node time)", "Bias (log est - log true)", "Share of nodes"]
        for ax, st, t in zip(axes[r], stats, titles):
            for colour, band in zip(colours, AGE_LABELS):
                ax.plot(st.index, st[band], "o-", c=colour, label=band)
            ax.set(title=f"{label(mode)}: {t}", xlabel=xlabel)
            if log_x:
                ax.set_xscale("log")
            ax.set_xticks(values)
            ax.set_xticklabels([f"{v:g}" for v in values])
            ax.minorticks_off()
        axes[r, 1].axhline(0, color="grey", ls=":")
        axes[r, 2].set_yscale("log")
    axes[0, 2].legend(title="true age (gens)", fontsize=7, loc="lower left")
    fig.suptitle(title_of(df, path))
    fig.tight_layout()
    fig.savefig(common.RESULTS / f"{path.stem}_age_bands.png", dpi=150)
    plt.close(fig)


def plot_accuracy(path):
    df = pd.read_csv(path, low_memory=False)
    name = path.stem
    modes = [m for m in ("true", "inferred") if m in set(df["mode"])]
    values = sorted(df["value"].unique())

    # Summary metrics versus the swept parameter
    summ = df.groupby(["mode", "value", "seed"]).apply(metrics, include_groups=False)
    summ = summ.groupby(["mode", "value"]).mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for k, mode in enumerate(modes):
        s = summ[summ["mode"] == mode]
        c = f"C{k}"
        axes[0].plot(s.value, s.rmse_log, "o-", c=c, label=label(mode))
        axes[1].plot(s.value, s.bias_log, "o-", c=c, label=label(mode))
        for lev, ls in ((0.5, "--"), (0.9, "-")):
            axes[2].plot(
                s.value,
                s[f"cov_{lev}"],
                "o",
                ls=ls,
                c=c,
                label=f"{label(mode)}, {lev:.0%} interval",
            )
    for lev in (0.5, 0.9):
        axes[2].axhline(lev, color="grey", ls=":")
    sweep = df.sweep.iloc[0]
    xlabel = "n (diploid samples)" if sweep == "sample_size" else sweep
    for ax, t in zip(
        axes,
        [
            "RMSE (log node time)",
            "Bias (log est - log true node time)",
            "Node time interval coverage",
        ],
    ):
        ax.set(xlabel=xlabel, title=t)
        if sweep != "max_iterations":
            ax.set_xscale("log")
        ax.set_xticks(values)
        ax.set_xticklabels([f"{v:g}" for v in values])
        ax.minorticks_off()
        ax.legend(fontsize=7)
    fig.suptitle(title_of(df, path))
    fig.tight_layout()
    fig.savefig(common.RESULTS / f"{name}_summary.png", dpi=150)
    plt.close(fig)
    plot_age_bands(df, path, modes, values, xlabel, sweep != "max_iterations")

    # Estimated vs true scatter: one row per swept value, one column per mode
    fig, axes = plt.subplots(
        len(values),
        len(modes),
        figsize=(3.8 * len(modes) + 1.5, 2.8 * len(values) + 0.8),
        squeeze=False,
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    points = None
    for j, mode in enumerate(modes):
        for i, value in enumerate(values):
            d = df[(df["mode"] == mode) & (df.value == value)]
            d = d[(d.true_time > 0) & (d.est_time > 0)]
            ax = axes[i, j]
            if mode == "inferred":
                d = d.sort_values("overlap", ascending=False)  # poor matches on top
                points = ax.scatter(
                    d.true_time,
                    d.est_time,
                    s=3,
                    alpha=0.5,
                    c=d.overlap,
                    cmap="viridis",
                    vmin=0,
                    vmax=1,
                )
            else:
                ax.scatter(d.true_time, d.est_time, s=3, alpha=0.3, c="C1")
            lim = [d.true_time.min(), d.true_time.max()]
            ax.plot(lim, lim, "k--", lw=0.8)
            ax.set(xscale="log", yscale="log")
            if len(d) > 2:
                r2 = np.corrcoef(np.log(d.true_time), np.log(d.est_time))[0, 1] ** 2
                rho = scipy.stats.spearmanr(d.true_time, d.est_time)[0]
                ax.text(
                    0.04,
                    0.96,
                    f"$R^2$ = {r2:.2f}\nSpearman $\\rho$ = {rho:.2f}",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=8,
                    bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.5),
                )
            ax.set_title(
                f"{label(mode)}, {param_label(df.sweep.iloc[0], value)}", fontsize=9
            )
    for ax in axes[-1]:
        ax.set_xlabel("true node time")
    for ax in axes[:, 0]:
        ax.set_ylabel("estimated node time")
    fig.suptitle(title_of(df, path))
    if points is not None:
        fig.colorbar(
            points,
            ax=axes,
            shrink=0.4,
            label="tsinferred topology: fraction of node's span\n"
            "matching the best-matched true node",
        )
    fig.savefig(common.RESULTS / f"{name}_scatter.png", dpi=150)
    plt.close(fig)

    # Calibration (PIT histograms)
    fig, axes = plt.subplots(
        1, len(modes), figsize=(4.5 * len(modes), 3.6), squeeze=False
    )
    for ax, mode in zip(axes[0], modes):
        d = df[(df["mode"] == mode) & (df.true_time > 0) & (df.post_var > 0)]
        ax.hist(gamma_cdf(d), bins=20, density=True)
        ax.axhline(1, color="k", ls="--", lw=0.8)
        ax.set(
            title=label(mode),
            xlabel="posterior CDF at true node time (PIT)\n"
            "0 = truth younger than posterior, 1 = truth older",
            ylabel="density of nodes",
        )
    fig.suptitle(
        f"Calibration (PIT): {title_of(df, path)}\n"
        "flat = well calibrated; U-shape = overconfident (intervals too narrow)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(common.RESULTS / f"{name}_calibration.png", dpi=150)
    plt.close(fig)
    summ.to_csv(common.RESULTS / f"{name}_metrics.csv", index=False)


def plot_performance(path):
    df = pd.read_csv(path)
    g = df.groupby(["length", "sample_size"]).median(numeric_only=True).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for length, s in g.groupby("length"):
        axes[0].plot(s.sample_size, s.time_s, "o-", label=f"L={length:g}")
        axes[1].plot(s.sample_size, s.peak_rss_mb, "o-", label=f"L={length:g}")
    for ax, t in zip(axes, ["Time (s)", "Peak RSS (MB)"]):
        ax.set(xscale="log", yscale="log", xlabel="sample size (diploids)", title=t)
        ax.legend()
    fig.suptitle(title_of(df, path))
    fig.tight_layout()
    fig.savefig(common.RESULTS / f"{path.stem}.png", dpi=150)
    plt.close(fig)


def main():
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    for path in sorted(common.RESULTS.glob("accuracy_*.csv")):
        if not path.stem.endswith("_metrics"):
            plot_accuracy(path)
            print("plotted", path)
    for perf in sorted(common.RESULTS.glob("performance*.csv")):
        plot_performance(perf)
        print("plotted", perf)


if __name__ == "__main__":
    main()
