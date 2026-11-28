"""
Equivalence of the phasing functions of ``_tsdate`` with their oracles: random
inputs built as in ``ExpectationPropagation``, degenerate inputs, and every error.
"""

import _tsdate
import numpy as np
import pytest
import tskit
from oracle import phasing
from oracle.errors import (
    BLOCK_OUT_OF_BOUNDS,
    PHASE_OUT_OF_RANGE,
    UNPHASED_INDIVIDUAL_EDGES,
    UNPHASED_LIKELIHOOD_NOT_CONSERVED,
)
from oracle.helpers import assert_same_error, assert_same_result, make_ts

from tsdate.phasing import block_singletons
from tsdate.rescaling import count_mutations


def block_singletons_kwargs(ts, unphased):
    return dict(
        individuals_unphased=np.array(unphased, dtype=bool),
        nodes_individual=ts.nodes_individual,
        mutations_node=ts.mutations_node,
        mutations_position=ts.sites_position[ts.mutations_site],
        edges_child=ts.edges_child,
        edges_left=ts.edges_left,
        edges_right=ts.edges_right,
        indexes_insert=ts.indexes_edge_insertion_order,
        indexes_remove=ts.indexes_edge_removal_order,
        sequence_length=ts.sequence_length,
    )


# nodes 0, 1 belong to individual 0, nodes 2, 3 to individual 1, node 4 is the root
NODES = [(1, 0, 0), (1, 0, 0), (1, 0, 1), (1, 0, 1), (0, 1, -1)]
STAR = [(0, 10, 4, 0), (0, 10, 4, 1), (0, 10, 4, 2), (0, 10, 4, 3)]
GAP = [(0, 4, 4, 0), (0, 4, 4, 1), (6, 10, 4, 0), (6, 10, 4, 1)] + STAR[2:]
SKEW = [(0, 4, 4, 0), (0, 6, 4, 1), (4, 10, 4, 0), (6, 10, 4, 1)] + STAR[2:]
SPLIT = [(0, 5, 4, 0), (5, 10, 4, 0), (0, 10, 4, 1)] + STAR[2:]
NO_INDIVIDUALS = [(1, 0, -1), (1, 0, -1), (0, 1, -1)]
ONE_INDIVIDUAL = [(1, 0, 0), (1, 0, 0), (1, 0, -1), (0, 1, -1)]

# nodes, number of individuals, edges, mutations, unphased
BLOCK_EDGE_CASES = [
    pytest.param(NODES, 2, STAR, [(1, 4)], [1, 1], id="no_singletons"),
    pytest.param(NODES, 2, STAR, [(1, 0)], [1, 1], id="singleton"),
    pytest.param(NODES, 2, STAR, [(1, 0)], [0, 0], id="all_phased"),
    pytest.param(NODES, 2, STAR, [], [1, 1], id="no_mutations"),
    pytest.param(NODES, 2, [], [(1, 0)], [1, 1], id="no_edges"),
    pytest.param(
        NO_INDIVIDUALS,
        0,
        [(0, 10, 2, 0), (0, 10, 2, 1)],
        [(1, 0)],
        [],
        id="no_individuals",
    ),
    pytest.param(
        ONE_INDIVIDUAL,
        1,
        [(0, 10, 3, 0), (0, 10, 3, 1), (0, 10, 3, 2)],
        [(1, 0), (2, 2)],
        [1],
        id="node_without_individual",
    ),
]

# edges, mutations, unphased, mutations_block (block ids are numbered in the order
# that blocks start; a mutation in no block is NULL)
BLOCK_NUMBERING = [
    pytest.param(STAR, [(1, 0)], [1, 1], [0], id="singleton"),
    pytest.param(STAR, [(1, 0), (2, 2), (3, 0)], [1, 0], [0, tskit.NULL, 0], id="mixed"),
    pytest.param(
        GAP,
        [(1, 0), (5, 0), (7, 0), (7.5, 1)],
        [1, 1],
        [0, tskit.NULL, 2, 2],
        id="mutation_in_gap",
    ),
    pytest.param(SPLIT, [(1, 0), (7, 0)], [1, 1], [0, 2], id="breakpoint_one_node"),
    pytest.param(
        SKEW,
        [(1, 0), (5, 0), (5.5, 1), (8, 1), (8.5, 0)],
        [1, 1],
        [0, 2, 2, 3, 3],
        id="breakpoints_differ",
    ),
]


class TestBlockSingletons:
    @pytest.mark.parametrize(
        ("nodes", "num_individuals", "edges", "mutations", "unphased"),
        BLOCK_EDGE_CASES,
    )
    def test_edge_cases(self, nodes, num_individuals, edges, mutations, unphased):
        ts = make_ts(nodes, edges, mutations, 10.0, num_individuals)
        kwargs = block_singletons_kwargs(ts, unphased)
        assert_same_result(_tsdate.block_singletons, phasing.block_singletons, **kwargs)

    def test_one_haplotype_edge_missing(self):
        # NOTE: current behaviour, possibly a bug: reachable from real data
        # (one haplotype of an unphased individual missing over a region); its block
        # is never flushed and the final consistency check fails
        edges = [(0, 10, 4, 0), (0, 10, 4, 2), (0, 10, 4, 3)]
        ts = make_ts(NODES, edges, [(1, 0)], 10.0, 2)
        kwargs = block_singletons_kwargs(ts, [1, 1])
        assert_same_error(
            _tsdate.block_singletons,
            phasing.block_singletons,
            UNPHASED_INDIVIDUAL_EDGES,
            **kwargs,
        )

    @pytest.mark.parametrize(
        ("edges", "mutations", "unphased", "expected"), BLOCK_NUMBERING
    )
    def test_block_numbering(self, edges, mutations, unphased, expected):
        ts = make_ts(NODES, edges, mutations, 10.0, 2)
        kwargs = block_singletons_kwargs(ts, unphased)
        _, _, mutations_block = assert_same_result(
            _tsdate.block_singletons, phasing.block_singletons, **kwargs
        )
        assert mutations_block.tolist() == expected

    @pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("seed", range(2))
    def test_random(self, small_ts_unphased, seed, fraction):
        ts = small_ts_unphased
        rng = np.random.default_rng(seed)
        unphased = rng.random(ts.num_individuals) < fraction
        kwargs = block_singletons_kwargs(ts, unphased)
        assert_same_result(_tsdate.block_singletons, phasing.block_singletons, **kwargs)


def reallocate_unphased_kwargs(ts):
    """
    Arguments of ``reallocate_unphased`` for ``ts``, with all individuals unphased and
    every phase one half
    """
    unphased = np.full(ts.num_individuals, True)
    _, blocks_edges, mutations_block = block_singletons(ts, unphased)
    edges_likelihood, _ = count_mutations(ts)
    return dict(
        edges_likelihood=edges_likelihood,
        mutations_phase=np.full(ts.num_mutations, 0.5),
        mutations_block=mutations_block,
        blocks_edges=blocks_edges,
    )


def first_blocked_mutation(kwargs):
    """
    The first mutation that is in a block
    """
    return np.flatnonzero(kwargs["mutations_block"] != tskit.NULL)[0]


# edges_likelihood, mutations_phase, mutations_block, blocks_edges
REALLOCATE_SMALL = [
    pytest.param(
        [[2.0, 5.0], [0.0, 5.0]],
        [0.0, 0.0],
        [0, 0],
        [[0, 1]],
        id="phases_zero",
    ),
    pytest.param(
        [[2.0, 5.0], [0.0, 5.0]],
        [1.0, 1.0],
        [0, 0],
        [[0, 1]],
        id="phases_one",
    ),
    pytest.param(
        [[2.0, 5.0], [0.0, 5.0], [7.0, 5.0]],
        [0.25, 1.0, np.nan],
        [0, 0, -1],
        [[0, 1]],
        id="single_block_with_unblocked_nan",
    ),
    pytest.param(
        [[3.0, 1.0], [1.0, 1.0], [0.0, 1.0]],
        [0.5, 0.5, 0.5, 0.5],
        [0, 0, 1, 1],
        [[0, 1], [1, 2]],
        id="edge_in_two_blocks",
    ),
    pytest.param(
        np.zeros((0, 2)),
        [],
        [],
        np.zeros((0, 2)),
        id="nothing_at_all",
    ),
]


class TestReallocateUnphased:
    @pytest.mark.parametrize(
        ("edges_likelihood", "mutations_phase", "mutations_block", "blocks_edges"),
        REALLOCATE_SMALL,
    )
    def test_small(
        self, edges_likelihood, mutations_phase, mutations_block, blocks_edges
    ):
        assert_same_result(
            _tsdate.reallocate_unphased,
            phasing.reallocate_unphased,
            edges_likelihood=np.array(edges_likelihood, dtype=np.float64),
            mutations_phase=np.array(mutations_phase, dtype=np.float64),
            mutations_block=np.array(mutations_block, dtype=np.int32),
            blocks_edges=np.array(blocks_edges, dtype=np.int32),
        )

    def test_null_edge_in_block(self):
        assert_same_error(
            _tsdate.reallocate_unphased,
            phasing.reallocate_unphased,
            BLOCK_OUT_OF_BOUNDS,
            edges_likelihood=np.ones((2, 2)),
            mutations_phase=np.array([0.5]),
            mutations_block=np.array([0], dtype=np.int32),
            blocks_edges=np.array([[0, -1]], dtype=np.int32),
        )

    @pytest.mark.parametrize("mode", ["fractional", "mixed"])
    @pytest.mark.parametrize("seed", range(3))
    def test_random(self, small_ts_unphased, seed, mode):
        ts = small_ts_unphased
        rng = np.random.default_rng(seed)
        kwargs = reallocate_unphased_kwargs(ts)
        if mode == "fractional":
            kwargs["mutations_phase"] = rng.uniform(0, 1, ts.num_mutations)
        else:
            phases = [0.0, 1.0, 0.5, 0.25]
            kwargs["mutations_phase"] = rng.choice(phases, ts.num_mutations)
        assert_same_result(
            _tsdate.reallocate_unphased, phasing.reallocate_unphased, **kwargs
        )

    @pytest.mark.parametrize("value", [1.5, -1e-9, np.inf, -np.inf])
    def test_phase_out_of_range(self, small_ts_unphased, value):
        kwargs = reallocate_unphased_kwargs(small_ts_unphased)
        kwargs["mutations_phase"][first_blocked_mutation(kwargs)] = value
        assert_same_error(
            _tsdate.reallocate_unphased,
            phasing.reallocate_unphased,
            PHASE_OUT_OF_RANGE,
            **kwargs,
        )

    @pytest.mark.parametrize("value", [np.nan, 7.0])
    def test_phase_ignored_without_block(self, small_ts_unphased, value):
        kwargs = reallocate_unphased_kwargs(small_ts_unphased)
        unblocked = kwargs["mutations_block"] == tskit.NULL
        kwargs["mutations_phase"][unblocked] = value
        assert_same_result(
            _tsdate.reallocate_unphased, phasing.reallocate_unphased, **kwargs
        )

    def test_nan_phase(self, small_ts_unphased):
        # NOTE: current behaviour, possibly a bug: a NaN phase on a
        # blocked mutation is skipped, leaving its singleton unallocated, so the
        # final `np.isclose` check fails. It only passes if the likelihoods happen to
        # total one less
        kwargs = reallocate_unphased_kwargs(small_ts_unphased)
        kwargs["mutations_phase"][first_blocked_mutation(kwargs)] = np.nan
        assert_same_error(
            _tsdate.reallocate_unphased,
            phasing.reallocate_unphased,
            UNPHASED_LIKELIHOOD_NOT_CONSERVED,
            **kwargs,
        )
        first_edge = kwargs["blocks_edges"][0, 0]
        kwargs["edges_likelihood"][first_edge, 0] -= 1.0
        assert_same_result(
            _tsdate.reallocate_unphased, phasing.reallocate_unphased, **kwargs
        )

    def test_no_mutations(self, small_ts_unphased):
        # likelihood on the edges of the blocks, but no mutation to allocate it
        kwargs = reallocate_unphased_kwargs(small_ts_unphased)
        kwargs["mutations_phase"] = np.zeros(0)
        kwargs["mutations_block"] = np.zeros(0, dtype=np.int32)
        assert_same_error(
            _tsdate.reallocate_unphased,
            phasing.reallocate_unphased,
            UNPHASED_LIKELIHOOD_NOT_CONSERVED,
            **kwargs,
        )

    def test_no_mutations_in_blocks(self, small_ts_unphased):
        # likelihood on the edges of the blocks, but no mutation to allocate it
        kwargs = reallocate_unphased_kwargs(small_ts_unphased)
        kwargs["mutations_block"][:] = tskit.NULL
        assert_same_error(
            _tsdate.reallocate_unphased,
            phasing.reallocate_unphased,
            UNPHASED_LIKELIHOOD_NOT_CONSERVED,
            **kwargs,
        )


def mutation_frequency_kwargs(ts, sample_sets):
    """
    Arguments of ``mutation_frequency`` for ``ts``, with a column of ``nodes_sample``
    for each list of nodes in ``sample_sets``
    """
    nodes_sample = np.zeros((ts.num_nodes, len(sample_sets)), dtype=bool)
    for i, nodes in enumerate(sample_sets):
        nodes_sample[list(nodes), i] = True
    return dict(
        nodes_sample=nodes_sample,
        mutations_node=ts.mutations_node,
        mutations_position=ts.sites_position[ts.mutations_site],
        edges_parent=ts.edges_parent,
        edges_child=ts.edges_child,
        edges_left=ts.edges_left,
        edges_right=ts.edges_right,
        indexes_insert=ts.indexes_edge_insertion_order,
        indexes_remove=ts.indexes_edge_removal_order,
        sequence_length=ts.sequence_length,
    )


# samples 0, 1, 2 (time 0), internal nodes 3 (time 1) and 4 (time 2); node 3 is a
# child of 4 throughout, but its parent edge is split at position 5
FREQ_NODES = [(1, 0), (1, 0), (1, 0), (0, 1), (0, 2)]
FREQ_EDGES = [(0, 10, 3, 0), (0, 10, 3, 1), (0, 5, 4, 3), (0, 10, 4, 2), (5, 10, 4, 3)]
FREQ_MUTATIONS = [(1, 0), (2, 3), (3, 4), (6, 2), (7, 3)]
FREQ_TS = make_ts(FREQ_NODES, FREQ_EDGES, FREQ_MUTATIONS, 10.0)

SAMPLE_SETS = [
    pytest.param([[0, 1, 2]], id="all_samples"),
    pytest.param([[0]], id="single_sample"),
    pytest.param([[]], id="empty_set"),
    pytest.param([], id="no_sets"),
    pytest.param([[], [0, 1, 2]], id="empty_and_full"),
    pytest.param([[0, 1], [1, 2], [0, 1, 2]], id="overlapping"),
    pytest.param([[0, 1], [0, 1]], id="identical"),
    pytest.param([[3]], id="internal_node"),
    pytest.param([[4, 0]], id="root_and_sample"),
]


class TestMutationFrequency:
    @pytest.mark.parametrize("sample_sets", SAMPLE_SETS)
    def test_sample_sets(self, sample_sets):
        kwargs = mutation_frequency_kwargs(FREQ_TS, sample_sets)
        assert_same_result(
            _tsdate.mutation_frequency, phasing.mutation_frequency, **kwargs
        )

    def test_no_mutations(self):
        ts = make_ts([(1, 0), (0, 1)], [(0, 10, 1, 0)], [], 10.0)
        kwargs = mutation_frequency_kwargs(ts, [[0]])
        assert_same_result(
            _tsdate.mutation_frequency, phasing.mutation_frequency, **kwargs
        )

    def test_no_edges(self):
        # NOTE: current behaviour, possibly a bug: with no edges the
        # sweep is never entered, so every frequency is zero, even above a member
        ts = make_ts([(1, 0), (0, 1)], [], [(1, 0), (2, 1)], 10.0)
        kwargs = mutation_frequency_kwargs(ts, [[0, 1]])
        assert_same_result(
            _tsdate.mutation_frequency, phasing.mutation_frequency, **kwargs
        )

    def test_mutation_beyond_last_edge(self):
        nodes = [(1, 0), (1, 0), (0, 1)]
        edges = [(0, 5, 2, 0), (0, 5, 2, 1)]
        ts = make_ts(nodes, edges, [(1, 2), (7, 2)], 10.0)
        kwargs = mutation_frequency_kwargs(ts, [[0, 1]])
        assert_same_result(
            _tsdate.mutation_frequency, phasing.mutation_frequency, **kwargs
        )

    def test_mutation_at_breakpoint(self):
        # a mutation exactly at a breakpoint belongs to the tree to its right
        ts = make_ts(FREQ_NODES, FREQ_EDGES, [(5, 4)], 10.0)
        kwargs = mutation_frequency_kwargs(ts, [[0, 1, 2]])
        assert_same_result(
            _tsdate.mutation_frequency, phasing.mutation_frequency, **kwargs
        )

    def test_unsorted_mutations(self):
        order = np.array([3, 0, 4, 1, 2])
        kwargs = mutation_frequency_kwargs(FREQ_TS, [[0, 1, 2]])
        kwargs["mutations_node"] = kwargs["mutations_node"][order]
        kwargs["mutations_position"] = kwargs["mutations_position"][order]
        assert_same_result(
            _tsdate.mutation_frequency, phasing.mutation_frequency, **kwargs
        )

    @pytest.mark.parametrize("num_sets", [1, 4])
    @pytest.mark.parametrize("seed", range(2))
    def test_random(self, small_ts, seed, num_sets):
        ts = small_ts
        rng = np.random.default_rng(seed)
        sample_sets = []
        for _ in range(num_sets):
            size = rng.integers(0, ts.num_samples + 1)
            sample_sets.append(rng.choice(ts.samples(), size=size, replace=False))
        kwargs = mutation_frequency_kwargs(ts, sample_sets)
        assert_same_result(
            _tsdate.mutation_frequency, phasing.mutation_frequency, **kwargs
        )
