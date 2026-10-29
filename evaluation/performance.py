"""
Time and memory of tsdate.variational_gamma. Each case runs in a fresh (spawned)
process so that memory is not polluted by earlier cases. Memory is the increase in
resident set size (RSS) during dating over the level just before it, sampled from
the parent process (the process itself would be starved by jitted code holding the
GIL). Writes results/performance.csv; use plot.py to make figures.

Example:
  python evaluation/performance.py --sizes 10 50 100 500 1000 --lengths 1e6 1e7
"""

import argparse
import multiprocessing
import os
import queue as queue_lib
import time

import common
import pandas as pd
import psutil
import tqdm

POLL_S = 0.01


def _case(queue, n, length, mutation_rate, seed, inferred):
    import tsdate

    ts = common.simulate(n, length, mutation_rate, seed)
    if inferred:
        ts = common.infer_topology(ts)
    # Warm up numba compilation on a tiny case so it isn't timed
    tsdate.date(
        common.simulate(5, 1e4, 1e-8, 1), mutation_rate=1e-8, method="variational_gamma"
    )
    queue.put(("start", None))
    start = time.perf_counter()
    tsdate.date(ts, mutation_rate=mutation_rate, method="variational_gamma")
    elapsed = time.perf_counter() - start
    queue.put(
        (
            "done",
            {
                "num_nodes": ts.num_nodes,
                "num_edges": ts.num_edges,
                "num_trees": ts.num_trees,
                "num_mutations": ts.num_mutations,
                "time_s": elapsed,
            },
        )
    )


def _rss_mb(ps):
    try:
        return ps.memory_info().rss / 1e6
    except psutil.NoSuchProcess:
        return None


def run_case(**kwargs):
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_case, args=(queue,), kwargs=kwargs)
    proc.start()
    ps = psutil.Process(proc.pid)
    baseline = peak = None
    while True:
        try:
            kind, result = queue.get(timeout=POLL_S)
        except queue_lib.Empty:
            rss = _rss_mb(ps)
            if rss is not None and baseline is not None:
                peak = max(peak, rss)
            continue
        if kind == "start":
            baseline = peak = _rss_mb(ps)
        else:
            break
    proc.join()
    return dict(
        result,
        baseline_rss_mb=baseline,
        peak_rss_mb=peak,
        extra_rss_mb=peak - baseline,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sizes", nargs="+", type=int, default=[10, 50, 100, 500, 1000])
    p.add_argument("--lengths", nargs="+", type=float, default=[1e6, 1e7])
    p.add_argument("--mutation-rate", type=float, default=1e-8)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--inferred", action="store_true", help="Use a tsinfer topology")
    p.add_argument(
        "--output",
        help="Output CSV name in results/ (default performance.csv); give each "
        "commit its own name so runs can be compared with compare.py",
    )
    p.add_argument("--quick", action="store_true", help="Tiny smoke-test sizes")
    args = p.parse_args()
    if args.quick:
        args.sizes, args.lengths, args.repeats = [10, 30], [1e5], 1

    # Compiled code is not part of the timing or the memory increase (it is warmed up
    # first), so cache it on disk to avoid recompiling in every fresh process
    os.environ.setdefault("TSDATE_ENABLE_NUMBA_CACHE", "1")
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
                    f"+{res['extra_rss_mb']:.0f} MB"
                )
                progress.update()
    progress.close()
    common.RESULTS.mkdir(exist_ok=True)
    suffix = "_quick" if args.quick else ""
    out = common.RESULTS / (args.output or f"performance{suffix}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
