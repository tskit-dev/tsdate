"""
Compare two performance runs (e.g. before and after a change) case by case.

    python evaluation/compare.py results/base.csv results/new.csv

Prints median time and memory increase for each (length, sample size) case, and the
ratio new/base (below 1 is an improvement). Differences within the run-to-run spread
of the repeats are marked "~" rather than as a change, so use --repeats >= 3 in
performance.py for this to be meaningful. See evaluation/README.md for the workflow.
"""

import argparse

import pandas as pd

KEYS = ["length", "sample_size", "inferred"]
METRICS = ["time_s", "extra_rss_mb"]


def summarise(path):
    df = pd.read_csv(path)
    g = df.groupby(KEYS)
    out = g[METRICS].median()
    for m in METRICS:
        out[m + "_spread"] = g[m].max() - g[m].min()
    label = ",".join(sorted(df.git_sha.dropna().astype(str).unique()))
    return out, label


def flag(ratio, base, new, base_spread, new_spread):
    if abs(new - base) <= max(base_spread, new_spread):
        return "~"
    return "faster/smaller" if ratio < 1 else "SLOWER/LARGER"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("base", help="performance CSV of the reference version")
    p.add_argument("new", help="performance CSV of the version being tested")
    args = p.parse_args()
    base, base_label = summarise(args.base)
    new, new_label = summarise(args.new)
    print(f"base: {args.base} ({base_label})\nnew:  {args.new} ({new_label})\n")
    both = base.join(new, how="inner", lsuffix="_b", rsuffix="_n")
    if both.empty:
        raise SystemExit("No cases in common between the two files")
    names = {"time_s": "time (s)", "extra_rss_mb": "memory increase (MB)"}
    for m in METRICS:
        print(names[m])
        print(f"{'length':>9} {'n':>6} {'base':>9} {'new':>9} {'new/base':>9}  verdict")
        for (length, n, _), r in both.iterrows():
            b, w = r[m + "_b"], r[m + "_n"]
            ratio = w / b if b else float("nan")
            verdict = flag(ratio, b, w, r[m + "_spread_b"], r[m + "_spread_n"])
            print(f"{length:>9.0e} {n:>6} {b:>9.2f} {w:>9.2f} {ratio:>9.2f}  {verdict}")
        print()


if __name__ == "__main__":
    main()
