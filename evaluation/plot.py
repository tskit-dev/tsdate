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
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

LEVELS = (0.5, 0.8, 0.9, 0.95)
MODE_LABELS = {"true": "true topology", "inferred": "tsinferred topology"}


# "auto" scores point estimates (scatter, RMSE, bias) against the edge midpoint, the
# best any method can do, but scores the posterior (PIT, interval coverage) against
# the exact simulated time, since the posterior is for the realised mutation time.
MUTATION_TRUTH = "auto"


def mutation_truth(purpose):
    """Truth used for mutations: 'midpoint' or 'exact'; purpose is 'error' or 'pit'."""
    if MUTATION_TRUTH == "auto":
        return "exact" if purpose == "pit" else "midpoint"
    return MUTATION_TRUTH


def read_results(path, purpose="error"):
    """Read a results CSV; for mutations, optionally use the exact simulated time."""
    df = pd.read_csv(path, dtype={"mode": str})
    if mutation_truth(purpose) == "exact" and "true_time_exact" in df:
        df["true_time"] = df["true_time_exact"]
    return df


def true_time_name(noun, purpose="error"):
    """e.g. 'true node time' or 'true mutation time (midpoint)'."""
    if noun == "node":
        return "true node time"
    return f"true mutation time ({mutation_truth(purpose)})"


def label(mode):
    return MODE_LABELS.get(mode, mode)


def param_label(sweep, value):
    return f"n={value:g}" if sweep == "sample_size" else f"{sweep}={value:g}"


def title_of(df, path, noun=None):
    quick = " (quick run, not representative)" if path.stem.endswith("_quick") else ""
    what = f"{noun} times, " if noun else ""
    return f"{what}tsdate method: {method_of(df)}{quick}"


def method_of(df):
    return df["method"].iloc[0] if "method" in df else "variational_gamma"


def gamma_cdf(df):
    """CDF of each moment-matched gamma posterior at the true time."""
    shape = df.post_mean**2 / df.post_var
    scale = df.post_var / df.post_mean
    return scipy.stats.gamma.cdf(df.true_time, a=shape, scale=scale)


def log_corr(x, y):
    """Pearson correlation of log values; NaN if undefined (too few or constant)."""
    lx, ly = np.log(x), np.log(y)
    if len(lx) < 3 or np.ptp(lx) == 0 or np.ptp(ly) == 0:
        return np.nan
    return np.corrcoef(lx, ly)[0, 1]


def metrics(df):
    ok = (df.true_time > 0) & (df.est_time > 0)
    df = df[ok]
    logerr = np.log(df.est_time) - np.log(df.true_time)
    pit = gamma_cdf(df)
    out = {
        "n": len(df),
        "rmse_log": np.sqrt(np.mean(logerr**2)),
        "bias_log": logerr.mean(),
        "corr_log": log_corr(df.est_time, df.true_time),
    }
    for lev in LEVELS:
        lo, hi = (1 - lev) / 2, 1 - (1 - lev) / 2
        out[f"cov_{lev}"] = np.mean((pit > lo) & (pit < hi))
    return pd.Series(out)


AGE_EDGES = [0, 10, 100, 1000, np.inf]
AGE_LABELS = ["<10", "10-100", "100-1000", ">1000"]
# Colour-blind-safe (Okabe-Ito based) scheme: true topology is green; tsinferred is
# orange/red, with the overlap colourmap running from vermillion ("bad") to blue.
_cmap = matplotlib.colors.LinearSegmentedColormap.from_list
MODE_CMAPS = {
    "true": _cmap("true", ["#B2E2D0", "#009E73", "#00573F"]),
    "inferred": _cmap("inferred", ["#FDBE85", "#D55E00", "#8C2D04"]),
}
MODE_POINT_COLOURS = {"true": "#009E73", "inferred": "#D55E00"}
OVERLAP_CMAP = _cmap("overlap", ["#D55E00", "#FFFFBF", "#0072B2"])


def band_shades(mode):
    """One shade per age band: darker = older nodes."""
    return MODE_CMAPS[mode](np.linspace(0.1, 1.0, len(AGE_LABELS)))


def age_band_summary(df):
    d = df.copy()
    d["band"] = pd.cut(d.true_time, AGE_EDGES, labels=AGE_LABELS, right=False)
    return (
        d.groupby(["mode", "band", "value", "seed"], observed=True)
        .apply(metrics, include_groups=False)
        .groupby(["mode", "band", "value"], observed=True)
        .mean()
        .reset_index()
    )


def plot_summary(path, mut_path=None):
    """RMSE, bias and coverage vs the swept value: node row, then mutation row."""
    inputs = [(path, "node")]
    if mut_path is not None and mut_path.exists():
        inputs.insert(0, (mut_path, "mutation"))
    fig, axes = plt.subplots(
        len(inputs),
        3,
        figsize=(15, 3.8 * len(inputs)),
        layout="constrained",
        squeeze=False,
    )
    for row, (p, noun) in zip(axes, inputs):
        df = read_results(p)
        summ = age_band_summary(df)
        cov_summ = age_band_summary(read_results(p, "pit"))
        values = sorted(df["value"].unique())
        sweep = df.sweep.iloc[0]
        xlabel = "n (diploid samples)" if sweep == "sample_size" else sweep
        for mode in [m for m in ("true", "inferred") if m in set(df["mode"])]:
            for colour, band in zip(band_shades(mode), AGE_LABELS):
                for ax, col in zip(row, ["rmse_log", "bias_log", "cov_0.9"]):
                    src = cov_summ if col == "cov_0.9" else summ
                    s = src[(src["mode"] == mode) & (src.band == band)]
                    ax.plot(
                        s.value,
                        s[col],
                        "o-",
                        c=colour,
                        ms=4,
                        label=f"{label(mode)}\n{band} generations",
                    )
        row[1].axhline(0, color="grey", ls=":")
        row[2].axhline(0.9, color="grey", ls=":")
        truth = true_time_name(noun)
        cov_truth = true_time_name(noun, "pit")
        for ax, t in zip(
            row,
            [
                f"RMSE (log time: tsdate vs {truth})",
                f"Bias (log tsdate - log {truth})",
                f"90% interval coverage (vs {cov_truth})",
            ],
        ):
            ax.set(xlabel=xlabel, title=t)
            if sweep != "max_iterations":
                ax.set_xscale("log")
            ax.set_xticks(values)
            ax.set_xticklabels([f"{v:g}" for v in values])
            ax.minorticks_off()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside center right",
        fontsize=7,
        title="topology, true age (gens)",
        title_fontsize=7,
    )
    fig.suptitle(title_of(df, path))
    fig.savefig(common.RESULTS / f"{path.stem}_summary.png", dpi=150)
    plt.close(fig)


def plot_scatter(path, mut_path=None):
    """Estimated vs true: columns are swept values; rows are node/mutation x mode."""
    inputs = [(path, "node")]
    if mut_path is not None and mut_path.exists():
        inputs.insert(0, (mut_path, "mutation"))
    frames = [(read_results(p), noun) for p, noun in inputs]
    df0 = frames[0][0]
    values = sorted(df0["value"].unique())
    sweep = df0.sweep.iloc[0]
    rows = [
        (df, noun, mode)
        for df, noun in frames
        for mode in ("true", "inferred")
        if mode in set(df["mode"])
    ]
    fig, axes = plt.subplots(
        len(rows),
        len(values),
        figsize=(2.2 * len(values) + 0.5, 2.9 * len(rows) + 0.8),
        squeeze=False,
        sharex="row",
        sharey="row",
        layout="constrained",
    )
    points = None
    for i, (df, noun, mode) in enumerate(rows):
        for j, value in enumerate(values):
            ax = axes[i, j]
            d = df[(df["mode"] == mode) & (df.value == value)]
            d = d[(d.true_time > 0) & (d.est_time > 0)]
            if len(d) == 0:
                ax.set_visible(False)
                continue
            if mode == "inferred" and "overlap" in d:
                d = d.sort_values("overlap", ascending=False)  # poor matches on top
                points = ax.scatter(
                    d.true_time,
                    d.est_time,
                    s=2,
                    alpha=0.15,
                    c=d.overlap,
                    cmap=OVERLAP_CMAP,
                    vmin=0,
                    vmax=1,
                    rasterized=True,
                )
            else:
                ax.scatter(
                    d.true_time,
                    d.est_time,
                    s=2,
                    alpha=0.15,
                    c=MODE_POINT_COLOURS[mode],
                    rasterized=True,
                )
            lim = [d.true_time.min(), d.true_time.max()]
            ax.plot(lim, lim, "k--", lw=0.8)
            ax.set(xscale="log", yscale="log")
            if len(d) > 2:
                r2 = log_corr(d.true_time, d.est_time) ** 2
                rho = scipy.stats.spearmanr(d.true_time, d.est_time)[0]
                ax.text(
                    0.04,
                    0.96,
                    f"$R^2$ = {r2:.2f}\n$\\rho$ = {rho:.2f}",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=7,
                    bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.5),
                )
            if i == 0:
                ax.set_title(param_label(sweep, value), fontsize=9)
        axes[i, 0].set_ylabel(f"Tsdate {noun} times\n({label(mode)})", fontsize=8)
        for ax in axes[i]:
            ax.set_xlabel(true_time_name(noun), fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle(
        title_of(df0, path) + "\n(\u03c1 = Spearman rank correlation)", fontsize=10
    )
    if points is not None:
        fig.colorbar(
            points,
            ax=axes,
            orientation="horizontal",
            location="bottom",
            shrink=0.4,
            aspect=40,
            label="tsinferred topology: fraction of node's span "
            "matching the best-matched true node",
        )
    fig.savefig(common.RESULTS / f"{path.stem}_scatter.png", dpi=150)
    plt.close(fig)


def plot_calibration(path, mut_path=None):
    """PIT histograms: a row for node times, then one for mutation times."""
    inputs = [(path, "node")]
    if mut_path is not None and mut_path.exists():
        inputs.insert(0, (mut_path, "mutation"))
    frames = [(read_results(p, "pit"), noun) for p, noun in inputs]
    modes = [m for m in ("true", "inferred") if m in set(frames[0][0]["mode"])]
    fig, axes = plt.subplots(
        len(frames),
        len(modes),
        figsize=(max(4.5 * len(modes), 8), 3.4 * len(frames) + 0.8),
        squeeze=False,
        layout="constrained",
    )
    for row, (df, noun) in zip(axes, frames):
        for ax, mode in zip(row, modes):
            d = df[(df["mode"] == mode) & (df.true_time > 0) & (df.post_var > 0)]
            ax.hist(gamma_cdf(d), bins=20, density=True, color=MODE_POINT_COLOURS[mode])
            ax.axhline(1, color="k", ls="--", lw=0.8)
            row_title = f"Tsdate {noun} times"
            if noun == "mutation":
                row_title += f" ({mutation_truth('pit')} truth)"
            ax.set(
                title=f"{row_title}\n({label(mode)})",
                xlabel=f"posterior CDF at {true_time_name(noun, 'pit')} (PIT)\n"
                "0 = truth younger than posterior, 1 = truth older",
                ylabel=f"density of {noun}s",
            )
    fig.suptitle(
        f"Calibration (PIT), {title_of(frames[0][0], path)}\n"
        "flat = well calibrated; U-shape = overconfident (intervals too narrow)",
        fontsize=10,
    )
    fig.savefig(common.RESULTS / f"{path.stem}_calibration.png", dpi=150)
    plt.close(fig)


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
    global MUTATION_TRUTH
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--mutation-truth",
        choices=["auto", "midpoint", "exact"],
        default="auto",
        help="true mutation time: midpoint of the true edge, or the exact simulated "
        "time. 'auto' (default) uses the midpoint for scatter, RMSE and bias, and "
        "the exact time for PIT and interval coverage",
    )
    MUTATION_TRUTH = parser.parse_args().mutation_truth
    for path in sorted(common.RESULTS.glob("accuracy_*.csv")):
        if path.stem.endswith("_metrics"):
            continue
        mut = path.with_name(path.name.replace("accuracy_", "mutations_", 1))
        plot_summary(path, mut)
        plot_scatter(path, mut)
        plot_calibration(path, mut)
        for p in (path, mut):
            if p.exists():
                summ = age_band_summary(read_results(p))
                # coverage is scored against the truth appropriate for the posterior
                cov = age_band_summary(read_results(p, "pit"))
                cov_cols = [c for c in cov if c.startswith("cov_")]
                summ = summ.drop(columns=cov_cols).merge(
                    cov[["mode", "band", "value", *cov_cols]],
                    on=["mode", "band", "value"],
                    how="left",
                )
                summ.to_csv(common.RESULTS / f"{p.stem}_metrics.csv", index=False)
        print("plotted", path)
    for perf in sorted(common.RESULTS.glob("performance*.csv")):
        plot_performance(perf)
        print("plotted", perf)


if __name__ == "__main__":
    main()
