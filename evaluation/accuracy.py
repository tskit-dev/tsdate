"""
Evaluate the accuracy of tsdate.variational_gamma on simulated data, sweeping one
parameter at a time. Writes one row per node to results/accuracy_<name>.csv and one
row per mutation to results/mutations_<name>.csv; use plot.py to make figures.

Modes:
  true  : date the true topology, compare node times to the truth
  inferred : date a tsinfer topology, compare each node's time to the time of the
             best-matching node in the true tree sequence (via tscompare), see
             https://github.com/tskit-dev/tsdate/issues/301

Example:
  python evaluation/accuracy.py --sweep sample_size --modes true inferred \\
      --seeds 3 --max-seeds 250 --cores 8
"""

import argparse
import concurrent.futures
import math
import multiprocessing

import common
import numpy as np
import pandas as pd
import tqdm
import tscompare
import tskit

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


def edge_midpoint_times(ts, mut_ids):
    """
    Time halfway between the child and parent node of each mutation's edge. A
    mutation's exact time is unknowable from the genotypes, so this is the best that
    any method can do. Mutations above a root (no parent) get NaN.
    """
    mut_ids = np.asarray(mut_ids)
    pos = ts.sites_position[ts.mutations_site[mut_ids]]
    nodes = ts.mutations_node[mut_ids]
    out = np.full(len(mut_ids), np.nan)
    tree = ts.first()
    for k in np.argsort(pos):
        tree.seek(pos[k])
        parent = tree.parent(nodes[k])
        if parent != tskit.NULL:
            out[k] = (ts.nodes_time[nodes[k]] + ts.nodes_time[parent]) / 2
    return out


def mutation_table(ts, true_ts):
    """
    Estimated (posterior mean) vs true mutation times, for sites that have exactly
    one mutation in both the dated ts and the true ts. Sites are matched by position,
    so this needs no node matching and works for any inferred topology. "true_time" is
    the midpoint of the true edge the mutation sits on; "true_time_exact" is the
    simulated time.
    """

    def by_position(t):
        per_site = np.bincount(t.mutations_site, minlength=t.num_sites)
        single = np.flatnonzero(per_site[t.mutations_site] == 1)
        return pd.Series(single, index=t.sites_position[t.mutations_site[single]])

    est, true = by_position(ts), by_position(true_ts)
    pos = est.index.intersection(true.index)
    est_id, true_id = est[pos].to_numpy(), true[pos].to_numpy()
    meta = [ts.mutation(i).metadata for i in est_id]
    mn = np.array([m["mn"] for m in meta])
    vr = np.array([m["vr"] for m in meta])
    true_time = edge_midpoint_times(true_ts, true_id)
    exact_time = true_ts.mutations_time[true_id]
    keep = (
        np.isfinite(true_time)
        & np.isfinite(exact_time)
        & np.isfinite(mn)
        & np.isfinite(vr)
    )
    return pd.DataFrame(
        {
            "position": pos[keep],
            "true_time": true_time[keep],
            "true_time_exact": exact_time[keep],
            "est_time": mn[keep],
            "post_mean": mn[keep],
            "post_var": vr[keep],
        }
    )


def run_case(args, value, seed, prov):
    """Simulate once, then date and score each mode. Returns (node, mutation) frames."""
    params = dict(DEFAULTS, max_iterations=None)
    params[args.sweep] = value
    true_ts = common.simulate(
        int(params["sample_size"]), params["length"], params["mutation_rate"], seed
    )
    frames, mut_frames = [], []
    for mode in args.modes:
        ts = true_ts if mode == "true" else common.infer_topology(true_ts)
        dated = tsdate.date(
            ts,
            method=METHOD,
            mutation_rate=params["mutation_rate"],
            max_iterations=params["max_iterations"],
            progress=False,
        )
        extra = dict(
            sweep=args.sweep, method=METHOD, value=value, seed=seed, mode=mode, **prov
        )
        frames.append(node_table(dated, true_ts, mode, args.min_overlap).assign(**extra))
        mut_frames.append(mutation_table(dated, true_ts).assign(**extra))
    return frames, mut_frames


def num_seeds(args, value, largest):
    """Replicates for a case: --seeds, or more for small n if --max-seeds is larger.

    Small samples have few young nodes per replicate, so they need more of them.
    """
    if args.sweep != "sample_size":
        return args.seeds
    return min(args.max_seeds, math.ceil(args.seeds * largest / value))


def run(args):
    sweeps = QUICK if args.quick else SWEEPS
    prov = common.provenance()
    values = sweeps[args.sweep]
    cases = [
        (v, s) for v in values for s in range(1, num_seeds(args, v, max(values)) + 1)
    ]
    # Larger values (more samples, longer sequence, higher mutation rate, more
    # iterations) take longer, so start them first to keep all cores busy to the end
    cases.sort(key=lambda case: -case[0])
    frames, mut_frames = [], []
    progress = tqdm.tqdm(total=len(cases), desc=args.sweep, unit="case")
    if args.cores == 1:
        for value, seed in cases:
            progress.set_postfix_str(f"{value:g}, seed {seed}")
            nodes, muts = run_case(args, value, seed, prov)
            frames += nodes
            mut_frames += muts
            progress.update()
    else:
        # Spawn (not fork) so that numba and tsinfer state is not shared
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(args.cores, mp_context=ctx) as ex:
            futures = [ex.submit(run_case, args, v, s, prov) for v, s in cases]
            for future in concurrent.futures.as_completed(futures):
                nodes, muts = future.result()
                frames += nodes
                mut_frames += muts
                progress.update()
    progress.close()
    common.RESULTS.mkdir(exist_ok=True)
    suffix = "_quick" if args.quick else ""
    out = common.RESULTS / f"accuracy_{args.sweep}{suffix}.csv"
    df = pd.concat(frames).sort_values(["mode", "value", "seed", "node"])
    df.to_csv(out, index=False)
    print("wrote", out)
    out = common.RESULTS / f"mutations_{args.sweep}{suffix}.csv"
    df = pd.concat(mut_frames).sort_values(["mode", "value", "seed", "position"])
    df.to_csv(out, index=False)
    print("wrote", out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sweep", choices=SWEEPS, required=True)
    p.add_argument("--modes", nargs="+", choices=["true", "inferred"], default=["true"])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument(
        "--max-seeds",
        type=int,
        help="For --sweep sample_size: use --seeds at the largest n, scaling up "
        "inversely with n to at most this many seeds. Default: --seeds (no scaling).",
    )
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
    args = p.parse_args()
    args.max_seeds = args.max_seeds or args.seeds
    run(args)


if __name__ == "__main__":
    main()
