"""
Parity of the phasing functions of ``_tsdate`` with their oracles: random
inputs built as in ``ExpectationPropagation``, degenerate inputs, and every
error.
"""

import _tsdate
import numpy as np
import pytest
import tskit
from oracle import phasing
from oracle.errors import (
    BLOCK_OUT_OF_RANGE,
    PHASE_OUT_OF_RANGE,
    UNPHASED_INDIVIDUAL_EDGES,
    UNPHASED_LIKELIHOOD_NOT_CONSERVED,
)
from oracle.helpers import INF, NAN, compare, make_ts, shrink

from tsdate.phasing import block_singletons
from tsdate.rescaling import count_mutations


def sweep_args(ts):
    """
    Tree sequence arguments shared by the sweep-line functions
    """
    return [
        ts.mutations_node,
        ts.sites_position[ts.mutations_site],
        ts.edges_parent,
        ts.edges_child,
        ts.edges_left,
        ts.edges_right,
        ts.indexes_edge_insertion_order,
        ts.indexes_edge_removal_order,
        ts.sequence_length,
    ]


# nodes 0, 1 belong to individual 0, nodes 2, 3 to individual 1, node 4 is the root
NODES = [(1, 0, 0), (1, 0, 0), (1, 0, 1), (1, 0, 1), (0, 1, -1)]
STAR = [(0, 10, 4, 0), (0, 10, 4, 1), (0, 10, 4, 2), (0, 10, 4, 3)]
GAP = [(0, 4, 4, 0), (0, 4, 4, 1), (6, 10, 4, 0), (6, 10, 4, 1)] + STAR[2:]
SKEW = [(0, 4, 4, 0), (0, 6, 4, 1), (4, 10, 4, 0), (6, 10, 4, 1)] + STAR[2:]
SPLIT = [(0, 5, 4, 0), (5, 10, 4, 0), (0, 10, 4, 1)] + STAR[2:]
NO_INDIVIDUALS = [(1, 0, -1), (1, 0, -1), (0, 1, -1)]
ONE_INDIVIDUAL = [(1, 0, 0), (1, 0, 0), (1, 0, -1), (0, 1, -1)]

# nodes, number of individuals, edges, mutations, unphased, exception of the oracle
BLOCK_EDGE_CASES = [
    pytest.param(NODES, 2, STAR, [(1, 4)], [1, 1], None, id="no_singletons"),
    pytest.param(NODES, 2, STAR, [(1, 0)], [1, 1], None, id="singleton"),
    pytest.param(NODES, 2, STAR, [(1, 0)], [0, 0], None, id="all_phased"),
    pytest.param(NODES, 2, STAR, [], [1, 1], None, id="no_mutations"),
    pytest.param(NODES, 2, [], [(1, 0)], [1, 1], None, id="no_edges"),
    pytest.param(
        NO_INDIVIDUALS,
        0,
        [(0, 10, 2, 0), (0, 10, 2, 1)],
        [(1, 0)],
        [],
        None,
        id="no_individuals",
    ),
    pytest.param(
        ONE_INDIVIDUAL,
        1,
        [(0, 10, 3, 0), (0, 10, 3, 1), (0, 10, 3, 2)],
        [(1, 0), (2, 2)],
        [1],
        None,
        id="node_without_individual",
    ),
    # NOTE: current behaviour, possibly a bug: reachable from real data
    # (one haplotype of an unphased individual missing over a region); its block is
    # never flushed and the final consistency check fails
    pytest.param(
        NODES,
        2,
        [(0, 10, 4, 0), (0, 10, 4, 2), (0, 10, 4, 3)],
        [(1, 0)],
        [1, 1],
        UNPHASED_INDIVIDUAL_EDGES,
        id="one_haplotype_edge_missing",
    ),
]

# edges, mutations, unphased, mutation_block (block ids are numbered in the order
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
    @staticmethod
    def args(ts, unphased):
        unphased = np.array(unphased, dtype=bool)
        return [unphased, ts.nodes_individual, *sweep_args(ts)]

    @staticmethod
    def compare(args):
        return compare(_tsdate.block_singletons, phasing.block_singletons, *args)

    @pytest.mark.parametrize(
        ("nodes", "num_individuals", "edges", "mutations", "unphased", "expected_error"),
        BLOCK_EDGE_CASES,
    )
    def test_edge_cases(
        self, nodes, num_individuals, edges, mutations, unphased, expected_error
    ):
        ts = make_ts(nodes, edges, mutations, 10.0, num_individuals)
        result = self.compare(self.args(ts, unphased))
        assert result.error == expected_error

    @pytest.mark.parametrize(
        ("edges", "mutations", "unphased", "expected"), BLOCK_NUMBERING
    )
    def test_block_numbering(self, edges, mutations, unphased, expected):
        ts = make_ts(NODES, edges, mutations, 10.0, 2)
        _, _, mutation_block = self.compare(self.args(ts, unphased)).value
        assert mutation_block.tolist() == expected

    @pytest.mark.parametrize(("index", "size"), [(5, 3), (8, 3), (3, 1)])
    def test_size_mismatch(self, index, size):
        ts = make_ts(NODES, STAR, [(1, 0), (2, 1)], 10.0, 2)
        args = shrink(self.args(ts, [1, 1]), index, size)
        result = self.compare(args)
        assert result.error is ValueError

    @pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("seed", range(2))
    def test_random(self, small_ts_unphased, seed, fraction):
        ts = small_ts_unphased
        rng = np.random.default_rng(seed)
        unphased = rng.random(ts.num_individuals) < fraction
        args = self.args(ts, unphased)
        self.compare(args)
        # also the dtypes, which `compare` ignores
        actual = _tsdate.block_singletons(*args)
        assert actual[1].dtype == actual[2].dtype == np.int32


@pytest.fixture(scope="module")
def blocks(small_ts_unphased):
    """
    Likelihoods and blocks for `reallocate_unphased`, with all individuals unphased
    """
    ts = small_ts_unphased
    unphased = np.full(ts.num_individuals, True)
    _, block_edges, mutation_blocks = block_singletons(ts, unphased)
    likelihoods, _ = count_mutations(ts)
    return ts, likelihoods, mutation_blocks, block_edges


# likelihoods, phase, mutation_blocks, blocks_edges, outcome of the oracle
REALLOCATE_SMALL = [
    pytest.param(
        np.array([[2.0, 5.0], [0.0, 5.0]]),
        np.array([0.0, 0.0]),
        np.array([0, 0], dtype=np.int32),
        np.array([[0, 1]], dtype=np.int32),
        None,
        id="phases_zero",
    ),
    pytest.param(
        np.array([[2.0, 5.0], [0.0, 5.0]]),
        np.array([1.0, 1.0]),
        np.array([0, 0], dtype=np.int32),
        np.array([[0, 1]], dtype=np.int32),
        None,
        id="phases_one",
    ),
    pytest.param(
        np.array([[2.0, 5.0], [0.0, 5.0], [7.0, 5.0]]),
        np.array([0.25, 1.0, NAN]),
        np.array([0, 0, -1], dtype=np.int32),
        np.array([[0, 1]], dtype=np.int32),
        None,
        id="single_block_with_unblocked_nan",
    ),
    pytest.param(
        np.array([[3.0, 1.0], [1.0, 1.0], [0.0, 1.0]]),
        np.full(4, 0.5),
        np.array([0, 0, 1, 1], dtype=np.int32),
        np.array([[0, 1], [1, 2]], dtype=np.int32),
        None,
        id="edge_in_two_blocks",
    ),
    pytest.param(
        np.zeros((0, 2)),
        np.zeros(0),
        np.zeros(0, dtype=np.int32),
        np.zeros((0, 2), dtype=np.int32),
        None,
        id="nothing_at_all",
    ),
    pytest.param(
        np.ones((2, 2)),
        np.array([0.5]),
        np.array([0], dtype=np.int32),
        np.array([[0, -1]], dtype=np.int32),
        BLOCK_OUT_OF_RANGE,
        id="null_edge_in_block",
    ),
]


class TestReallocateUnphased:
    @staticmethod
    def compare(*args):
        return compare(_tsdate.reallocate_unphased, phasing.reallocate_unphased, *args)

    @pytest.mark.parametrize(
        ("lik", "phase", "mutation_blocks", "blocks_edges", "expected_error"),
        REALLOCATE_SMALL,
    )
    def test_small(self, lik, phase, mutation_blocks, blocks_edges, expected_error):
        result = self.compare(lik, phase, mutation_blocks, blocks_edges)
        assert result.error == expected_error

    @pytest.mark.parametrize("mode", ["fractional", "mixed"])
    @pytest.mark.parametrize("seed", range(3))
    def test_random(self, blocks, seed, mode):
        ts, lik, mutation_blocks, block_edges = blocks
        rng = np.random.default_rng(seed)
        if mode == "fractional":
            phase = rng.uniform(0, 1, ts.num_mutations)
        else:
            phase = rng.choice([0.0, 1.0, 0.5, 0.25], ts.num_mutations)
        self.compare(lik, phase, mutation_blocks, block_edges)

    @pytest.mark.parametrize("value", [1.5, -1e-9, INF, -INF])
    def test_phase_out_of_range(self, blocks, value):
        _, lik, mutation_blocks, block_edges = blocks
        phase = np.full(mutation_blocks.size, 0.5)
        phase[np.flatnonzero(mutation_blocks >= 0)[0]] = value
        result = self.compare(lik, phase, mutation_blocks, block_edges)
        assert result.error == PHASE_OUT_OF_RANGE

    @pytest.mark.parametrize("value", [NAN, 7.0])
    def test_phase_ignored_without_block(self, blocks, value):
        _, lik, mutation_blocks, block_edges = blocks
        phase = np.full(mutation_blocks.size, 0.5)
        phase[mutation_blocks == tskit.NULL] = value
        result = self.compare(lik, phase, mutation_blocks, block_edges)
        assert result.error is None

    def test_nan_phase(self, blocks):
        # NOTE: current behaviour, possibly a bug: a NaN phase on a
        # blocked mutation is skipped, leaving its singleton unallocated, so the
        # final `np.isclose` check fails. It only passes if the likelihoods happen to
        # total one less
        _, lik, mutation_blocks, block_edges = blocks
        phase = np.full(mutation_blocks.size, 0.5)
        phase[np.flatnonzero(mutation_blocks >= 0)[0]] = NAN
        result = self.compare(lik, phase, mutation_blocks, block_edges)
        assert result.error == UNPHASED_LIKELIHOOD_NOT_CONSERVED
        lik = lik.copy()
        lik[block_edges[0, 0], 0] -= 1.0
        result = self.compare(lik, phase, mutation_blocks, block_edges)
        assert result.error is None

    @pytest.mark.parametrize("case", ["no_mutations", "all_unblocked"])
    def test_blocks_without_singletons(self, blocks, case):
        # likelihood mass on the block edges, but no mutation to allocate it
        ts, lik, _, block_edges = blocks
        size = 0 if case == "no_mutations" else ts.num_mutations
        phase = np.full(size, 0.5)
        mutation_blocks = np.full(size, tskit.NULL, dtype=np.int32)
        result = self.compare(lik, phase, mutation_blocks, block_edges)
        assert result.error == UNPHASED_LIKELIHOOD_NOT_CONSERVED

    @pytest.mark.parametrize("case", ["phase", "blocks_transposed"])
    def test_size_mismatch(self, blocks, case):
        ts, lik, mutation_blocks, block_edges = blocks
        phase = np.full(ts.num_mutations, 0.5)
        if case == "phase":
            phase = phase[:-1]
        else:
            block_edges = np.ascontiguousarray(block_edges.T)
        result = self.compare(lik, phase, mutation_blocks, block_edges)
        assert result.error is ValueError


# samples 0, 1, 2 (time 0), internal nodes 3 (time 1) and 4 (time 2); node 3 is a
# child of 4 throughout, but its parent edge is split at position 5
FREQ_NODES = [(1, 0), (1, 0), (1, 0), (0, 1), (0, 2)]
FREQ_EDGES = [(0, 10, 3, 0), (0, 10, 3, 1), (0, 5, 4, 3), (0, 10, 4, 2), (5, 10, 4, 3)]
FREQ_MUTATIONS = [(1, 0), (2, 3), (3, 4), (6, 2), (7, 3)]

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


@pytest.fixture(scope="module")
def freq_ts():
    return make_ts(FREQ_NODES, FREQ_EDGES, FREQ_MUTATIONS, 10.0)


class TestMutationFrequency:
    @staticmethod
    def compare(nodes_sample, args):
        return compare(
            _tsdate.mutation_frequency,
            phasing.mutation_frequency,
            nodes_sample,
            *args,
        )

    @staticmethod
    def nodes_sample(ts, members):
        nodes_sample = np.zeros((ts.num_nodes, len(members)), dtype=bool)
        for i, m in enumerate(members):
            nodes_sample[list(m), i] = True
        return nodes_sample

    @pytest.mark.parametrize("members", SAMPLE_SETS)
    def test_sample_sets(self, freq_ts, members):
        nodes_sample = self.nodes_sample(freq_ts, members)
        self.compare(nodes_sample, sweep_args(freq_ts))

    def test_no_mutations(self):
        ts = make_ts([(1, 0), (0, 1)], [(0, 10, 1, 0)], [], 10.0)
        self.compare(self.nodes_sample(ts, [[0]]), sweep_args(ts))

    def test_no_edges(self):
        # NOTE: current behaviour, possibly a bug: with no edges the
        # sweep is never entered, so every frequency is zero, even above a member
        ts = make_ts([(1, 0), (0, 1)], [], [(1, 0), (2, 1)], 10.0)
        self.compare(self.nodes_sample(ts, [[0, 1]]), sweep_args(ts))

    def test_mutation_beyond_last_edge(self):
        nodes = [(1, 0), (1, 0), (0, 1)]
        edges = [(0, 5, 2, 0), (0, 5, 2, 1)]
        ts = make_ts(nodes, edges, [(1, 2), (7, 2)], 10.0)
        self.compare(self.nodes_sample(ts, [[0, 1]]), sweep_args(ts))

    def test_mutation_at_breakpoint(self):
        # a mutation exactly at a breakpoint belongs to the tree to its right
        ts = make_ts(FREQ_NODES, FREQ_EDGES, [(5, 4)], 10.0)
        self.compare(self.nodes_sample(ts, [[0, 1, 2]]), sweep_args(ts))

    def test_unsorted_mutations(self, freq_ts):
        order = np.array([3, 0, 4, 1, 2])
        args = sweep_args(freq_ts)
        args[0], args[1] = args[0][order], args[1][order]
        self.compare(self.nodes_sample(freq_ts, [[0, 1, 2]]), args)

    @pytest.mark.parametrize(("index", "size"), [(3, 3), (6, 3), (1, 2)])
    def test_size_mismatch(self, freq_ts, index, size):
        nodes_sample = np.ones((freq_ts.num_nodes, 1), dtype=bool)
        args = shrink(sweep_args(freq_ts), index, size)
        result = self.compare(nodes_sample, args)
        assert result.error is ValueError

    @pytest.mark.parametrize("num_sets", [1, 4])
    @pytest.mark.parametrize("seed", range(2))
    def test_random(self, small_ts, seed, num_sets):
        ts = small_ts
        rng = np.random.default_rng(seed)
        members = [
            rng.choice(
                ts.samples(), size=int(rng.integers(0, ts.num_samples + 1)), replace=False
            )
            for _ in range(num_sets)
        ]
        nodes_sample = self.nodes_sample(ts, members)
        self.compare(nodes_sample, sweep_args(ts))
        # also the dtype, which `compare` ignores
        actual = _tsdate.mutation_frequency(nodes_sample, *sweep_args(ts))
        assert actual.dtype == np.int32
