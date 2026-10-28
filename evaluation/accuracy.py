"""
Evaluate the accuracy of tsdate.variational_gamma on simulated data, sweeping one
parameter at a time. Writes one row per node to results/accuracy_<name>.csv; use
plot.py to make figures.

Modes:
  true  : date the true topology, compare node times to the truth
  inferred : date a tsinfer topology, compare each node's time to the time of the
             best-matching node in the true tree sequence (via tscompare), see
             https://github.com/tskit-dev/tsdate/issues/301

Example:
  python evaluation/accuracy.py --sweep sample_size --modes true inferred \\
      --seeds 3 --cores 4
"""

import argparse
import concurrent.futures
import multiprocessing

import common
import numpy as np
import pandas as pd
import tqdm
import tscompare

import tsdate

METHOD = "variational_gamma"
DEFAULTS = {"sample_size": 100, "length": 1e6, "mutation_rate": 1e-8}
SWEEPS = {
    "sample_size": [10, 50, 200, 1000, 2000, 5000],
    "length": [5e4, 5e5, 5e6],
    "mutation_rate": [1e-9, 1e-8, 1e-7],
    "max_iterations": [1, 5, 25],
}
QUICK = {
    "sample_size": [10, 30],
    "length": [1e5, 3e5],
    "mutation_rate": [1e-8],
    "max_iterations": [2, 5],
}


def node_table(ts, true_ts, mode, overlap_threshold):
    """Estimated vs true times for non-sample nodes in the dated ts."""
    is_internal = ~np.isin(np.arange(ts.num_nodes), ts.samples())
    if mode == "true":
        true_time = true_ts.nodes_time.copy()
        overlap = np.ones(ts.num_nodes)
        matched = np.arange(ts.num_nodes)
    else:
        true_time, shared_span, matched = tscompare.match_node_ages(ts, true_ts)
        # match_node_ages returns the shared span in bp, not a proportion
        node_span = tscompare.node_spans(ts)
        overlap = np.divide(
            shared_span, node_span, out=np.zeros(ts.num_nodes), where=node_span > 0
        )
    keep = is_internal & (overlap >= overlap_threshold) & np.isfinite(true_time)
    mn = np.array([n.metadata["mn"] for n in ts.nodes()])
    vr = np.array([n.metadata["vr"] for n in ts.nodes()])
    return pd.DataFrame(
        {
            "node": np.flatnonzero(keep),
            "matched_true_node": matched[keep],
            "true_time": true_time[keep],
            "est_time": ts.nodes_time[keep],
            "post_mean": mn[keep],
            "post_var": vr[keep],
            "overlap": overlap[keep],
        }
    )


def run_case(args, value, seed, prov):
    """Simulate once, then date and score each mode. Returns a list of DataFrames."""
    params = dict(DEFAULTS, max_iterations=None)
    params[args.sweep] = value
    true_ts = common.simulate(
        int(params["sample_size"]), params["length"], params["mutation_rate"], seed
    )
    frames = []
    for mode in args.modes:
        ts = true_ts if mode == "true" else common.infer_topology(true_ts)
        dated = tsdate.date(
            ts,
            method=METHOD,
            mutation_rate=params["mutation_rate"],
            max_iterations=params["max_iterations"],
            progress=False,
        )
        df = node_table(dated, true_ts, mode, args.min_overlap)
        frames.append(
            df.assign(
                sweep=args.sweep,
                method=METHOD,
                value=value,
                seed=seed,
                mode=mode,
                **prov,
            )
        )
    return frames


def run(args):
    sweeps = QUICK if args.quick else SWEEPS
    prov = common.provenance()
    cases = [(v, s) for v in sweeps[args.sweep] for s in range(1, args.seeds + 1)]
    frames = []
    progress = tqdm.tqdm(total=len(cases), desc=args.sweep, unit="case")
    if args.cores == 1:
        for value, seed in cases:
            progress.set_postfix_str(f"{value:g}, seed {seed}")
            frames += run_case(args, value, seed, prov)
            progress.update()
    else:
        # Spawn (not fork) so that numba and tsinfer state is not shared
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(args.cores, mp_context=ctx) as ex:
            futures = [ex.submit(run_case, args, v, s, prov) for v, s in cases]
            for future in concurrent.futures.as_completed(futures):
                frames += future.result()
                progress.update()
    progress.close()
    common.RESULTS.mkdir(exist_ok=True)
    suffix = "_quick" if args.quick else ""
    out = common.RESULTS / f"accuracy_{args.sweep}{suffix}.csv"
    df = pd.concat(frames).sort_values(["mode", "value", "seed", "node"])
    df.to_csv(out, index=False)
    print("wrote", out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sweep", choices=SWEEPS, required=True)
    p.add_argument("--modes", nargs="+", choices=["true", "inferred"], default=["true"])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument(
        "--min-overlap",
        type=float,
        default=0.0,
        help="Minimum matched-span fraction for inferred nodes",
    )
    p.add_argument(
        "--cores",
        type=int,
        default=1,
        help="Number of worker processes (cases are independent). Note that memory "
        "use is roughly cores x the largest case.",
    )
    p.add_argument("--quick", action="store_true", help="Tiny smoke-test sizes")
    run(p.parse_args())


if __name__ == "__main__":
    main()
