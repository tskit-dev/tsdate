"""
Microbenchmarks of the functions of the `_tsdate` extension in tsdate. Inputs are built
from simulated tree sequences at several sizes, mirroring how the functions are called
in production. Scalar functions (approx, hypergeo) are not timed on their own: they are
exercised through the EP passes. Writes a table to stdout and, with --output, a CSV to
results/.

Timing: each function is warmed up (one untimed call), then called `--repeats` times on
identical inputs (a new EP state for each call of an EP pass, made outside the timed
region); the median of `time.perf_counter` is reported. Very fast calls are batched, so
that a timed block lasts at least a few ms (reported times are per call).

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


def _iterated_state(ep, iterations, max_shape):
    """
    A new ``_tsdate.ExpectationPropagation`` with the inputs of `ep` (a
    ``tsdate.variational.ExpectationPropagation``), after `iterations` iterations
    as in ``ExpectationPropagation.iterate``
    """
    import _tsdate

    from tsdate.variational import (
        LEAFWARD,
        ROOTWARD,
        USE_BLOCK_LIKELIHOOD,
        USE_EDGE_LIKELIHOOD,
    )

    state = _tsdate.ExpectationPropagation(
        constraints=ep.node_constraints,
        edges_parent=ep.edge_parents,
        edges_child=ep.edge_children,
        edges_likelihood=ep.edge_likelihoods,
        blocks_left=ep.block_nodes[ROOTWARD],
        blocks_right=ep.block_nodes[LEAFWARD],
        blocks_likelihood=ep.block_likelihoods,
    )
    for _ in range(iterations):
        for order, unphased in [
            (ep.block_order, USE_BLOCK_LIKELIHOOD),
            (ep.edge_order, USE_EDGE_LIKELIHOOD),
        ]:
            state.propagate_likelihood(
                order=order, unphased=unphased, max_shape=max_shape, min_step=0.1
            )
        state.propagate_prior(
            free=ep.unconstrained_roots, max_shape=max_shape, em_maxitt=10, em_reltol=1e-8
        )
        state.rescale_factors()
    return state


@dataclasses.dataclass
class Bench:
    """
    One benchmark: `call(**make_kwargs())`, where `make_kwargs()` returns fresh
    keyword arguments (with new copies of anything that the call changes).
    """

    name: str
    call: callable
    make_kwargs: callable
    note: str = ""


def _const(**kwargs):
    return lambda: kwargs


def _module_function(name):
    import _tsdate

    return getattr(_tsdate, name)


def _sweep_kwargs(ts, parents=True):
    """
    The edge, index and mutation arrays of `ts` as keyword arguments
    """
    kwargs = dict(
        mutations_node=ts.mutations_node,
        mutations_position=ts.sites_position[ts.mutations_site],
        edges_parent=ts.edges_parent,
        edges_child=ts.edges_child,
        edges_left=ts.edges_left,
        edges_right=ts.edges_right,
        indexes_insert=ts.indexes_edge_insertion_order,
        indexes_remove=ts.indexes_edge_removal_order,
    )
    if not parents:
        del kwargs["edges_parent"]
    return kwargs


def build_benchmarks(n, length, iterations):
    """
    Build all benchmarks (and a dict describing the inputs) for one size.
    """
    from tsdate import rescaling
    from tsdate.variational import USE_BLOCK_LIKELIHOOD, USE_EDGE_LIKELIHOOD

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
    func = _module_function

    # --- variational.py, as in ExpectationPropagation.iterate/infer ---
    # The passes update the state, so each call gets a new state, iterated outside
    # the timed region.
    max_shape, min_step = 1000, 0.1

    def propagate_likelihood(state, **kwargs):
        state.propagate_likelihood(**kwargs)

    def propagate_prior(state, **kwargs):
        state.propagate_prior(**kwargs)

    def propagate_mutations(state, **kwargs):
        state.propagate_mutations(**kwargs)

    for tag, ep, order, unphased in [
        ("edges", ep_old, ep_old.edge_order, USE_EDGE_LIKELIHOOD),
        ("blocks", ep_flat, ep_flat.block_order, USE_BLOCK_LIKELIHOOD),
    ]:

        def likelihood_kwargs(ep=ep, order=order, unphased=unphased):
            return dict(
                state=_iterated_state(ep, iterations, max_shape),
                order=order,
                unphased=unphased,
                max_shape=max_shape,
                min_step=min_step,
            )

        add(
            Bench(f"propagate_likelihood[{tag}]", propagate_likelihood, likelihood_kwargs)
        )

    def prior_kwargs():
        return dict(
            state=_iterated_state(ep_old, iterations, max_shape),
            free=ep_old.unconstrained_roots,
            max_shape=max_shape,
            em_maxitt=10,
            em_reltol=1e-8,
        )

    add(Bench("propagate_prior", propagate_prior, prior_kwargs))

    # propagate_mutations does not change the state
    for tag, ep, unphased in [
        ("edges", ep_old, USE_EDGE_LIKELIHOOD),
        ("blocks", ep_flat, USE_BLOCK_LIKELIHOOD),
    ]:
        blocked = ep.mutation_blocks != tskit.NULL
        if unphased:
            mutations_edge = ep.mutation_blocks[blocked]
        else:
            mutations_edge = ep.mutation_edges[~blocked]
        add(
            Bench(
                f"propagate_mutations[{tag}]",
                propagate_mutations,
                _const(state=ep._ep, mutations_edge=mutations_edge, unphased=unphased),
            )
        )

    # --- phasing.py ---
    reallocate_kwargs = dict(
        edges_likelihood=ep_flat.sizebiased_likelihoods,
        mutations_phase=ep_flat.mutation_phase,
        mutations_block=ep_flat.mutation_blocks,
        blocks_edges=ep_flat.block_edges,
    )
    add(
        Bench(
            "reallocate_unphased",
            func("reallocate_unphased"),
            _const(**reallocate_kwargs),
        )
    )
    add(
        Bench(
            "block_singletons",
            func("block_singletons"),
            _const(
                individuals_unphased=np.full(ts_flat.num_individuals, True),
                nodes_individual=ts_flat.nodes_individual,
                **_sweep_kwargs(ts_flat, parents=False),
                sequence_length=ts_flat.sequence_length,
            ),
        )
    )
    nodes_sample = np.full(ts_flat.num_nodes, False)
    nodes_sample[list(ts_flat.samples())] = True
    add(
        Bench(
            "mutation_frequency",
            func("mutation_frequency"),
            _const(
                nodes_sample=nodes_sample[:, np.newaxis],
                **_sweep_kwargs(ts_flat),
                sequence_length=ts_flat.sequence_length,
            ),
        )
    )

    # --- rescaling.py, as in ExpectationPropagation.rescale ---
    rescale_lik = func("reallocate_unphased")(**reallocate_kwargs)
    nodes_fixed = ep_flat.node_constraints[:, 0] == ep_flat.node_constraints[:, 1]
    nodes_time, _ = ep_flat.node_moments()
    edges = dict(edges_parent=ep_flat.edge_parents, edges_child=ep_flat.edge_children)
    intervals = 1000
    add(
        Bench(
            "mutational_area",
            func("mutational_area"),
            _const(nodes_time=nodes_time, edges_likelihood=rescale_lik, **edges),
        )
    )
    timescale_kwargs = dict(
        nodes_time=nodes_time,
        edges_likelihood=rescale_lik,
        nodes_fixed=nodes_fixed,
        **edges,
        max_intervals=intervals,
    )
    add(
        Bench(
            "mutational_timescale",
            func("mutational_timescale"),
            _const(**timescale_kwargs),
        )
    )
    original_breaks, rescaled_breaks = rescaling.mutational_timescale(**timescale_kwargs)
    add(
        Bench(
            "piecewise_scale_point_estimate",
            func("piecewise_scale_point_estimate"),
            _const(
                point_estimate=nodes_time,
                point_fixed=nodes_fixed,
                original_breaks=original_breaks,
                rescaled_breaks=rescaled_breaks,
            ),
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
            func("piecewise_scale_posterior"),
            _const(
                posteriors=ep_flat.node_posterior,
                posteriors_fixed=nodes_fixed,
                original_breaks=orig,
                rescaled_breaks=rescaled_breaks,
                quantile_width=0.5,
                max_shape=max_shape,
            ),
        )
    )
    add(
        Bench(
            "count_mutations",
            func("count_mutations"),
            _const(
                nodes_sample=nodes_sample,
                **_sweep_kwargs(ts_flat),
                sequence_length=ts_flat.sequence_length,
                size_biased=True,
            ),
        )
    )

    # --- util.py ---
    tree = dict(
        edges_parent=ts_flat.edges_parent,
        edges_child=ts_flat.edges_child,
        edges_left=ts_flat.edges_left,
        edges_right=ts_flat.edges_right,
    )
    split_kwargs = dict(**tree, nodes_excluded=nodes_sample)
    add(
        Bench(
            "split_disjoint_nodes",
            func("split_disjoint_nodes"),
            _const(**split_kwargs),
        )
    )
    new_parent, new_child, nodes_order, split_nodes = func("split_disjoint_nodes")(
        **split_kwargs
    )
    info["num_split_nodes"] = len(split_nodes)
    sweep = _sweep_kwargs(ts_flat)
    sweep.update(edges_parent=new_parent, edges_child=new_child)
    add(
        Bench(
            "relabel_mutations_node",
            func("relabel_mutations_node"),
            _const(nodes_order=nodes_order, **sweep),
        )
    )
    rng = np.random.default_rng(1)
    noisy_time = ts_flat.nodes_time * np.exp(rng.normal(0, 0.3, ts_flat.num_nodes))
    noisy_time[nodes_sample] = ts_flat.nodes_time[nodes_sample]
    add(
        Bench(
            "constrain_ages",
            func("constrain_ages"),
            _const(
                nodes_time=noisy_time,
                nodes_fixed=nodes_sample,
                edges_parent=ts_flat.edges_parent,
                edges_child=ts_flat.edges_child,
                epsilon=1e-6,
                max_iterations=20,
            ),
            note="noisy times violate constraints; 20 iterations",
        )
    )
    add(
        Bench(
            "contains_unary_nodes",
            func("contains_unary_nodes"),
            _const(
                nodes_mask=nodes_sample,
                edges_parent=ts_flat.edges_parent,
                edges_left=ts_flat.edges_left,
                edges_right=ts_flat.edges_right,
                indexes_insert=ts_flat.indexes_edge_insertion_order,
                indexes_remove=ts_flat.indexes_edge_removal_order,
                sequence_length=ts_flat.sequence_length,
            ),
        )
    )

    # --- prior.py ---
    add(
        Bench(
            "conditional_coalescent_variance",
            func("conditional_coalescent_variance"),
            _const(num_tips=ts_flat.num_samples),
        )
    )

    return benches, info


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def time_bench(bench, repeats):
    """
    Warm up, then return per-call times (seconds) for `repeats` timed blocks.
    """
    t0 = time.perf_counter()
    bench.call(**bench.make_kwargs())  # warm up (also the first-call cost)
    first_call = time.perf_counter() - t0
    # estimate the cost of one call to choose a batch size
    kwargs = bench.make_kwargs()
    t0 = time.perf_counter()
    bench.call(**kwargs)
    one = time.perf_counter() - t0
    batch = int(min(MAX_BATCH, max(1, math.ceil(MIN_BLOCK_S / max(one, 1e-9)))))
    times = []
    for _ in range(repeats):
        batch_kwargs = [bench.make_kwargs() for _ in range(batch)]
        t0 = time.perf_counter()
        for kwargs in batch_kwargs:
            bench.call(**kwargs)
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
            times, first, batch = time_bench(bench, args.repeats)
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
