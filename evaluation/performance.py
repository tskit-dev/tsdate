"""
Time and peak memory of tsdate.variational_gamma. Each case runs in a fresh
(spawned) process so that peak RSS is not polluted by earlier cases. Writes
results/performance.csv; use plot.py to make figures.

Example:
  python evaluation/performance.py --sizes 10 50 100 500 1000 --lengths 1e6 1e7
"""

import argparse
import multiprocessing
import resource
import sys
import time

import common
import pandas as pd
import tqdm


def _peak_rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e6 if sys.platform == "darwin" else rss / 1e3  # bytes vs KB


def _case(queue, n, length, mutation_rate, seed, inferred):
    import tsdate

    ts = common.simulate(n, length, mutation_rate, seed)
    if inferred:
        ts = common.infer_topology(ts)
    # Warm up numba compilation on a tiny case so it isn't timed
    tsdate.date(
        common.simulate(5, 1e4, 1e-8, 1), mutation_rate=1e-8, method="variational_gamma"
    )
    baseline = _peak_rss_mb()
    start = time.perf_counter()
    tsdate.date(ts, mutation_rate=mutation_rate, method="variational_gamma")
    elapsed = time.perf_counter() - start
    queue.put(
        {
            "num_nodes": ts.num_nodes,
            "num_edges": ts.num_edges,
            "num_trees": ts.num_trees,
            "num_mutations": ts.num_mutations,
            "time_s": elapsed,
            "peak_rss_mb": _peak_rss_mb(),
            "baseline_rss_mb": baseline,
        }
    )


def run_case(**kwargs):
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_case, args=(queue,), kwargs=kwargs)
    proc.start()
    result = queue.get()
    proc.join()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sizes", nargs="+", type=int, default=[10, 50, 100, 500, 1000])
    p.add_argument("--lengths", nargs="+", type=float, default=[1e6, 1e7])
    p.add_argument("--mutation-rate", type=float, default=1e-8)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--inferred", action="store_true", help="Use a tsinfer topology")
    p.add_argument("--quick", action="store_true", help="Tiny smoke-test sizes")
    args = p.parse_args()
    if args.quick:
        args.sizes, args.lengths, args.repeats = [10, 30], [1e5], 1

    prov = common.provenance()
    rows = []
    total = len(args.sizes) * len(args.lengths) * args.repeats
    progress = tqdm.tqdm(total=total, desc="performance", unit="run")
    for n in args.sizes:
        for length in args.lengths:
            for seed in range(1, args.repeats + 1):
                progress.set_postfix_str(f"n={n}, L={length:g}, seed {seed}")
                res = run_case(
                    n=n,
                    length=length,
                    mutation_rate=args.mutation_rate,
                    seed=seed,
                    inferred=args.inferred,
                )
                rows.append(
                    dict(
                        sample_size=n,
                        length=length,
                        seed=seed,
                        inferred=args.inferred,
                        method="variational_gamma",
                        **res,
                        **prov,
                    )
                )
                tqdm.tqdm.write(
                    f"n={n} L={length:g} seed={seed}: {res['time_s']:.2f}s, "
                    f"{res['peak_rss_mb']:.0f} MB"
                )
                progress.update()
    progress.close()
    common.RESULTS.mkdir(exist_ok=True)
    suffix = "_quick" if args.quick else ""
    out = common.RESULTS / f"performance{suffix}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
