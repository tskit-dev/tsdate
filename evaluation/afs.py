"""
Compute site and branch allele frequency spectra for a simulated tree sequence and
for its tsdate-dated version. Within each genomic window, the expected number of
sites at each frequency (branch AFS x mutation rate) should match the observed
number; a mismatch indicates bias in the dated branch lengths. Frequencies are
pooled into bins. Writes results/afs_<sample_size>.csv with one row per window and
bin; use plot.py to make figures.

Example:
  python evaluation/afs.py --sample-size 100 --length 1e7
"""

import argparse
from math import isqrt

import common
import numpy as np
import pandas as pd

import tsdate

METHOD = "variational_gamma"


def allele_frequency_spectra(
    ts,
    mutation_rate,
    max_freq=None,
    num_bins=9,
    num_windows=500,
    polarised=True,
    size_biased=False,
):
    """
    Observed (site) and expected (branch x mutation rate) AFS in each window, with
    adjacent frequencies pooled into roughly `num_bins` bins (a square number).
    Returns a long-format DataFrame with one row per window and bin.
    """
    if max_freq is None:
        max_freq = -1
    ts_trim = ts.trim()
    windows = np.linspace(0, ts_trim.sequence_length, num_windows + 1)
    if size_biased:
        bin_sizes = np.tile(np.arange(ts.num_samples + 1), (num_windows, 1))
    else:
        bin_sizes = np.ones((num_windows, ts.num_samples + 1))
    site_afs = bin_sizes * ts_trim.allele_frequency_spectrum(
        mode="site", windows=windows, span_normalise=False, polarised=polarised
    )
    site_afs = site_afs[:, 1:max_freq]
    branch_afs = (
        mutation_rate
        * bin_sizes
        * ts_trim.allele_frequency_spectrum(
            mode="branch", windows=windows, span_normalise=False, polarised=polarised
        )
    )
    branch_afs = branch_afs[:, 1:max_freq]
    num_bins = isqrt(num_bins) ** 2
    cumulative = np.arange(0, branch_afs.shape[1], dtype=np.float64)
    cumulative /= cumulative[-1]
    edges = np.linspace(0, 1, num_bins + 1)
    edges = np.searchsorted(cumulative, edges, side="right") - 1
    frames = []
    for b, (i, j) in enumerate(zip(edges[:-1], edges[1:])):
        frames.append(
            pd.DataFrame(
                {
                    "window": np.arange(num_windows),
                    "bin": b,
                    "freq_lo": i + 1,
                    "freq_hi": j + 1,
                    "observed": site_afs[:, i:j].sum(axis=1),
                    "expected": branch_afs[:, i:j].sum(axis=1),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sample-size", type=int, default=100, help="diploid samples")
    p.add_argument("--length", type=float, default=1e7)
    p.add_argument("--mutation-rate", type=float, default=1e-8)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    ts = common.simulate(args.sample_size, args.length, args.mutation_rate, args.seed)
    dated = tsdate.date(
        ts, method=METHOD, mutation_rate=args.mutation_rate, progress=False
    )
    prov = common.provenance()
    frames = [
        allele_frequency_spectra(t, args.mutation_rate).assign(
            times=name,
            method=METHOD,
            sample_size=args.sample_size,
            length=args.length,
            mutation_rate=args.mutation_rate,
            seed=args.seed,
            **prov,
        )
        for name, t in (("true", ts), ("tsdate", dated))
    ]
    common.RESULTS.mkdir(exist_ok=True)
    out = common.RESULTS / f"afs_{args.sample_size}.csv"
    pd.concat(frames).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
