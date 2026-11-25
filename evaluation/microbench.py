"""
Microbenchmarks of the array-level entry points of the `_tsdate` extension in tsdate
(the functions that take and return arrays). Inputs are built from simulated
tree sequences at several sizes, mirroring how the functions are called in
production. Scalar functions (approx, hypergeo) are not timed on their own: they are
exercised through the EP passes. Writes a table to stdout and, with --output, a CSV
to results/.

Timing: each function is warmed up (one untimed call), then called `--repeats` times
on identical, freshly copied inputs (copies are made outside the timed region); the
median of `time.perf_counter` is reported. Very fast calls are batched, so that a
timed block lasts at least a few ms (reported times are per call).

Example:
  python evaluation/microbench.py --repeats 7 --output microbench.csv
"""

import argparse
import dataclasses
import math
import statistics
import time

import common
import numpy as np
import pandas as pd
import tskit

# size name -> (diploid individuals, sequence length)
SIZES = {
    "small": (20, 1e5),
    "medium": (100, 1e6),
    "large": (1000, 5e7),
}
QUICK_SIZES = {"small": (10, 1e5)}
MUTATION_RATE = 1e-8
MIN_BLOCK_S = 5e-3  # batch fast calls until a timed block lasts this long
MAX_BATCH = 1000


def _with_factors(func, position):
    """Wrap a `_tsdate` function so that it takes an ``EPFactors`` at `position`."""

    def wrapped(*args):
        factors = args[position].arrays()
        return func(*args[:position], factors, *args[position + 1 :])

    return wrapped


def _functions():
    """The benchmarked functions of `_tsdate`, by the name the benchmarks use."""
    import _tsdate

    return {
        "propagate_likelihood": _with_factors(_tsdate.propagate_likelihood, 6),
        "propagate_prior": _with_factors(_tsdate.propagate_prior, 2),
        "propagate_mutations": _with_factors(_tsdate.propagate_mutations, 9),
        "_block_singletons": _tsdate.block_singletons,
        "reallocate_unphased": _tsdate.reallocate_unphased,
        "_mutation_frequency": _tsdate.mutation_frequency,
        "_split_disjoint_nodes": _tsdate.split_disjoint_nodes,
        "_relabel_mutations_node": _tsdate.relabel_mutations_node,
        "_constrain_ages": _tsdate.constrain_ages,
        "_contains_unary_nodes": _tsdate.contains_unary_nodes,
        "_marginalize_over_ancestors": _tsdate.marginalize_over_ancestors,
        "conditional_coalescent_variance": _tsdate.conditional_coalescent_variance,
        # rescaling
        "mutational_area": _tsdate.mutational_area,
        "mutational_timescale": _tsdate.mutational_timescale,
        "piecewise_scale_point_estimate": _tsdate.piecewise_scale_point_estimate,
        "piecewise_scale_posterior": _tsdate.piecewise_scale_posterior,
        "_fixed_changepoints": _tsdate.fixed_changepoints,
        "_poisson_changepoints": _tsdate.poisson_changepoints,
        "_count_mutations": _tsdate.count_mutations,
    }


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _with_ancient_samples(ts, fraction=0.1, seed=1):
    """
    Give some leaf samples nonzero ages (halfway to their youngest parent), so
    that the "fixed child, free parent" branch of the EP passes is exercised.
    """
    rng = np.random.default_rng(seed)
    tables = ts.dump_tables()
    time_ = tables.nodes.time
    youngest_parent = np.full(ts.num_nodes, np.inf)
    np.minimum.at(youngest_parent, ts.edges_child, ts.nodes_time[ts.edges_parent])
    is_leaf = ~np.isin(np.arange(ts.num_nodes), ts.edges_parent)
    candidates = [u for u in ts.samples() if is_leaf[u] and youngest_parent[u] > 1]
    chosen = rng.choice(candidates, size=int(fraction * len(candidates)), replace=False)
    time_[chosen] = youngest_parent[chosen] / 2
    tables.nodes.time = time_
    tables.mutations.time = np.full(ts.num_mutations, tskit.UNKNOWN_TIME)
    return tables.tree_sequence()


def _copy_factors(f):
    from tsdate.variational import EPFactors

    g = EPFactors(
        np.zeros((f.scale.size, 2)), f._p, f._c, f._j, f._k
    )  # constraints only give the shape
    g.node[:] = f.node
    g.edge[:] = f.edge
    g.block[:] = f.block
    g.scale[:] = f.scale
    return g


def _run_ep(ts, singletons_phased, iterations):
    from tsdate.variational import ExpectationPropagation

    ep = ExpectationPropagation(
        ts, mutation_rate=MUTATION_RATE, singletons_phased=singletons_phased
    )
    ep.infer(
        ep_iterations=iterations,
        max_shape=1000,
        rescale_intervals=0,  # call the rescaling functions separately
        rescale_iterations=0,
        regularise=True,
        rescale_segsites=False,
    )
    return ep


@dataclasses.dataclass
class Bench:
    """
    One benchmark: `make_args()` returns a fresh argument tuple (copying anything
    mutated in place) for the function called `func` in the implementation.
    """

    name: str
    func: str
    make_args: callable
    note: str = ""


def _const(*args):
    return lambda: args


def build_benchmarks(n, length, iterations):
    """Build all benchmarks (and a dict describing the inputs) for one size."""
    from tsdate import phasing, rescaling, util
    from tsdate.variational import (
        LEAFWARD,
        ROOTWARD,
        USE_BLOCK_LIKELIHOOD,
        USE_EDGE_LIKELIHOOD,
    )

    ts_flat = common.simulate(n, length, MUTATION_RATE, seed=1)  # contemporary
    ts_old = _with_ancient_samples(ts_flat)  # some samples with nonzero age
    ep_old = _run_ep(ts_old, True, iterations)  # edges
    ep_flat = _run_ep(ts_flat, False, iterations)  # unphased singleton blocks
    info = dict(
        num_samples=ts_flat.num_samples,
        num_nodes=ts_flat.num_nodes,
        num_edges=ts_flat.num_edges,
        num_mutations=ts_flat.num_mutations,
        num_blocks=ep_flat.block_likelihoods.shape[0],
    )
    benches = []
    add = benches.append

    # --- variational.py, as in ExpectationPropagation.iterate/infer ---
    max_shape, min_step = 1000, 0.1
    for tag, ep, order, par, chi, lik, logc, unph in [
        (
            "edges",
            ep_old,
            ep_old.edge_order,
            ep_old.edge_parents,
            ep_old.edge_children,
            ep_old.edge_likelihoods,
            ep_old.edge_logconst,
            USE_EDGE_LIKELIHOOD,
        ),
        (
            "blocks",
            ep_flat,
            ep_flat.block_order,
            ep_flat.block_nodes[ROOTWARD],
            ep_flat.block_nodes[LEAFWARD],
            ep_flat.block_likelihoods,
            ep_flat.block_logconst,
            USE_BLOCK_LIKELIHOOD,
        ),
    ]:

        def likelihood_args(
            ep=ep, order=order, par=par, chi=chi, lik=lik, logc=logc, unph=unph
        ):
            return (
                order,
                par,
                chi,
                lik,
                ep.node_constraints,
                ep.node_posterior.copy(),
                _copy_factors(ep.factors),
                logc.copy(),
                max_shape,
                min_step,
                unph,
            )

        add(
            Bench(f"propagate_likelihood[{tag}]", "propagate_likelihood", likelihood_args)
        )

    def prior_args():
        return (
            ep_old.unconstrained_roots,
            ep_old.node_posterior.copy(),
            _copy_factors(ep_old.factors),
            max_shape,
            10,
            1e-8,
        )

    add(Bench("propagate_prior", "propagate_prior", prior_args))

    for tag, ep, unph in [
        ("edges", ep_old, USE_EDGE_LIKELIHOOD),
        ("blocks", ep_flat, USE_BLOCK_LIKELIHOOD),
    ]:
        phased = ep.mutation_blocks == tskit.NULL
        if unph:
            sel = ~phased
            m_edges, par, chi = (
                ep.mutation_blocks,
                ep.block_nodes[ROOTWARD],
                ep.block_nodes[LEAFWARD],
            )
            lik = ep.block_likelihoods
        else:
            sel = phased
            m_edges, par, chi = ep.mutation_edges, ep.edge_parents, ep.edge_children
            lik = ep.edge_likelihoods

        def mutation_args(
            ep=ep, sel=sel, m_edges=m_edges, par=par, chi=chi, lik=lik, unph=unph
        ):
            return (
                ep.mutation_order[sel],
                ep.mutation_posterior.copy(),
                ep.mutation_phase.copy(),
                m_edges,
                par,
                chi,
                lik,
                ep.node_constraints,
                ep.node_posterior,
                ep.factors,
                unph,
            )

        add(Bench(f"propagate_mutations[{tag}]", "propagate_mutations", mutation_args))

    # --- phasing.py ---
    likelihoods = ep_flat.sizebiased_likelihoods
    add(
        Bench(
            "reallocate_unphased",
            "reallocate_unphased",
            lambda: (
                likelihoods.copy(),
                ep_flat.mutation_phase,
                ep_flat.mutation_blocks,
                ep_flat.block_edges,
            ),
        )
    )
    tree_args = (
        ts_flat.edges_parent,
        ts_flat.edges_child,
        ts_flat.edges_left,
        ts_flat.edges_right,
    )
    index_args = (
        ts_flat.indexes_edge_insertion_order,
        ts_flat.indexes_edge_removal_order,
    )
    muts_pos = ts_flat.sites_position[ts_flat.mutations_site]
    nodes_individual = ts_flat.nodes_individual
    add(
        Bench(
            "_block_singletons",
            "_block_singletons",
            _const(
                np.full(ts_flat.num_individuals, True),
                nodes_individual,
                ts_flat.mutations_node,
                muts_pos,
                *tree_args,
                *index_args,
                ts_flat.sequence_length,
            ),
        )
    )
    nodes_sample = np.full((ts_flat.num_nodes, 1), False)
    nodes_sample[list(ts_flat.samples()), 0] = True
    add(
        Bench(
            "_mutation_frequency",
            "_mutation_frequency",
            _const(
                nodes_sample,
                ts_flat.mutations_node,
                muts_pos,
                *tree_args,
                *index_args,
                ts_flat.sequence_length,
            ),
        )
    )

    node_is_sample = np.full(ts_flat.num_nodes, False)
    node_is_sample[list(ts_flat.samples())] = True

    # --- rescaling.py, as in ExpectationPropagation.rescale ---
    rescale_lik = ep_flat.sizebiased_likelihoods.copy()
    phasing.reallocate_unphased(
        rescale_lik,
        ep_flat.mutation_phase,
        ep_flat.mutation_blocks,
        ep_flat.block_edges,
    )
    nodes_fixed = ep_flat.node_constraints[:, 0] == ep_flat.node_constraints[:, 1]
    nodes_time, _ = ep_flat.node_moments()
    par, chi = ep_flat.edge_parents, ep_flat.edge_children
    intervals = 1000
    add(
        Bench(
            "mutational_area",
            "mutational_area",
            _const(nodes_time, rescale_lik, par, chi),
        )
    )
    add(
        Bench(
            "mutational_timescale",
            "mutational_timescale",
            _const(nodes_time, rescale_lik, nodes_fixed, par, chi, intervals),
        )
    )
    original_breaks, rescaled_breaks = rescaling.mutational_timescale(
        nodes_time, rescale_lik, nodes_fixed, par, chi, intervals
    )
    add(
        Bench(
            "piecewise_scale_point_estimate",
            "piecewise_scale_point_estimate",
            _const(nodes_time, nodes_fixed, original_breaks, rescaled_breaks),
        )
    )
    rescaled_time = rescaling.piecewise_scale_point_estimate(
        nodes_time, nodes_fixed, original_breaks, rescaled_breaks
    )
    _, unique = np.unique(rescaled_time[~nodes_fixed], return_index=True)
    orig = rescaling.piecewise_scale_point_estimate(
        rescaled_breaks,
        np.full(rescaled_breaks.size, False),
        np.append(0, rescaled_time[~nodes_fixed][unique]),
        np.append(0, nodes_time[~nodes_fixed][unique]),
    )
    add(
        Bench(
            "piecewise_scale_posterior",
            "piecewise_scale_posterior",
            _const(
                ep_flat.node_posterior, nodes_fixed, orig, rescaled_breaks, 0.5, max_shape
            ),
        )
    )
    counts, offset, duration, _ = rescaling.mutational_area(
        nodes_time, rescale_lik, par, chi
    )
    add(
        Bench(
            "_fixed_changepoints",
            "_fixed_changepoints",
            _const(offset * duration, intervals),
        )
    )
    # Unused in production (only referenced in a TODO); cap its size as it is slow.
    cap = 2000
    add(
        Bench(
            "_poisson_changepoints",
            "_poisson_changepoints",
            _const(counts[:cap], offset[:cap], 2.0, 0.0, 0.0),
            note=f"unused in production; first {cap} intervals",
        )
    )
    add(
        Bench(
            "_count_mutations",
            "_count_mutations",
            _const(
                node_is_sample,
                ts_flat.mutations_node,
                muts_pos,
                *tree_args,
                *index_args,
                ts_flat.sequence_length,
                True,
            ),
        )
    )

    # --- util.py ---
    split_args = (*tree_args, node_is_sample)
    add(Bench("_split_disjoint_nodes", "_split_disjoint_nodes", _const(*split_args)))
    new_parent, new_child, nodes_order, split_nodes = util._split_disjoint_nodes(
        *split_args
    )
    info["num_split_nodes"] = len(split_nodes)
    add(
        Bench(
            "_relabel_mutations_node",
            "_relabel_mutations_node",
            _const(
                ts_flat.mutations_node,
                muts_pos,
                nodes_order,
                new_parent,
                new_child,
                ts_flat.edges_left,
                ts_flat.edges_right,
                *index_args,
            ),
        )
    )
    rng = np.random.default_rng(1)
    noisy_time = ts_flat.nodes_time * np.exp(rng.normal(0, 0.3, ts_flat.num_nodes))
    noisy_time[node_is_sample] = ts_flat.nodes_time[node_is_sample]
    add(
        Bench(
            "_constrain_ages",
            "_constrain_ages",
            _const(
                noisy_time,
                node_is_sample,
                ts_flat.edges_parent,
                ts_flat.edges_child,
                1e-6,
                20,
            ),
            note="noisy times violate constraints; 20 iterations",
        )
    )
    add(
        Bench(
            "_contains_unary_nodes",
            "_contains_unary_nodes",
            _const(
                node_is_sample,
                ts_flat.edges_parent,
                ts_flat.edges_left,
                ts_flat.edges_right,
                *index_args,
                ts_flat.sequence_length,
                ts_flat.num_nodes,
            ),
        )
    )

    # --- prior.py (prior tables are built for up to a few thousand tips) ---
    num_tips = ts_flat.num_samples
    coal_rates = np.array(
        [2 / (i * (i - 1)) if i > 1 else 0.0 for i in range(1, num_tips + 1)]
    )
    moments = np.stack((coal_rates, 2 * coal_rates**2), 1)  # mean, second moment
    add(
        Bench(
            "_marginalize_over_ancestors",
            "_marginalize_over_ancestors",
            _const(moments),
        )
    )
    add(
        Bench(
            "conditional_coalescent_variance",
            "conditional_coalescent_variance",
            _const(np.uint64(num_tips)),
        )
    )

    return benches, info


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def time_bench(func, bench, repeats):
    """Warm up, then return per-call times (seconds) for `repeats` timed blocks."""
    t0 = time.perf_counter()
    func(*bench.make_args())  # warm up (also the first-call cost)
    first_call = time.perf_counter() - t0
    # estimate the cost of one call to choose a batch size
    args = bench.make_args()
    t0 = time.perf_counter()
    func(*args)
    one = time.perf_counter() - t0
    batch = int(min(MAX_BATCH, max(1, math.ceil(MIN_BLOCK_S / max(one, 1e-9)))))
    times = []
    for _ in range(repeats):
        batch_args = [bench.make_args() for _ in range(batch)]
        t0 = time.perf_counter()
        for a in batch_args:
            func(*a)
        times.append((time.perf_counter() - t0) / batch)
    return times, first_call, batch


def format_table(rows):
    cols = ["function", "size", "median_s", "min_s", "max_s", "repeats"]
    body = [
        [
            r["function"],
            r["size"],
            f"{r['median_s']:.3e}",
            f"{r['min_s']:.3e}",
            f"{r['max_s']:.3e}",
            str(r["repeats"]),
        ]
        for r in rows
    ]
    widths = [max(len(c), *(len(b[i]) for b in body)) for i, c in enumerate(cols)]
    lines = ["  ".join(c.ljust(w) for c, w in zip(cols, widths))]
    for b in body:
        lines.append("  ".join(x.ljust(w) for x, w in zip(b, widths)))
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sizes", nargs="+", default=list(SIZES), choices=list(SIZES))
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--ep-iterations", type=int, default=2, help="EP iterations first")
    p.add_argument(
        "--only", nargs="+", default=None, help="Substrings of benchmark names to run"
    )
    p.add_argument(
        "--output",
        help="Output CSV name in results/ (default: none, print only)",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Tiny smoke-test run",
    )
    args = p.parse_args()
    sizes = {k: SIZES[k] for k in args.sizes}
    if args.quick:
        sizes, args.repeats, args.ep_iterations = QUICK_SIZES, 1, 1

    funcs = _functions()

    prov = common.provenance()
    rows = []
    for size, (n, length) in sizes.items():
        t0 = time.perf_counter()
        benches, info = build_benchmarks(n, length, args.ep_iterations)
        print(
            f"[{size}] n={n} L={length:g}: {info} "
            f"(inputs built in {time.perf_counter() - t0:.1f} s)",
            flush=True,
        )
        for bench in benches:
            if args.only and not any(s in bench.name for s in args.only):
                continue
            times, first, batch = time_bench(funcs[bench.func], bench, args.repeats)
            row = dict(
                function=bench.name,
                size=size,
                median_s=statistics.median(times),
                min_s=min(times),
                max_s=max(times),
                repeats=args.repeats,
                batch=batch,
                first_call_s=first,
                sample_size=n,
                length=length,
                note=bench.note,
                **info,
                **prov,
            )
            rows.append(row)
            print(
                f"  {bench.name:36s} {row['median_s']:.3e} s (x{batch} per block)",
                flush=True,
            )

    print()
    print(format_table(rows))
    if args.output:
        common.RESULTS.mkdir(exist_ok=True)
        out = common.RESULTS / args.output
        pd.DataFrame(rows).to_csv(out, index=False)
        print("wrote", out)


if __name__ == "__main__":
    main()
