"""
Parity of the tree sequence utilities of ``_tsdate`` with their oracles: simulated tree
sequences, degenerate inputs, and every error.
"""

import _tsdate
import numpy as np
import pytest
import tskit
from oracle import util
from oracle.errors import FIXED_NODES_VIOLATE_CONSTRAINT, MUTATION_NODE_NOT_IN_TREE
from oracle.helpers import NAN, assert_outcome, b1, compare, f64, i32, make_ts, shrink


@pytest.fixture(params=["small_ts", "small_ts_historical"])
def sim_ts(request):
    return request.getfixturevalue(request.param)


def sample_flags(ts):
    return np.bitwise_and(ts.nodes_flags, tskit.NODE_IS_SAMPLE).astype(bool)


# parent, child, left, right, excluded, split nodes (new nodes are numbered from the
# number of nodes, and each is a copy of an old node: `split_nodes` lists the old ids)
SPLIT_CASES = [
    pytest.param([2, 2], [0, 1], [0, 0], [10, 10], [1, 1, 0], [], id="no_disjoint"),
    pytest.param([], [], [], [], [1, 0], [], id="no_edges"),
    pytest.param([], [], [], [], [], [], id="no_nodes"),
    pytest.param(
        [2, 2, 2], [0, 1, 1], [0, 2, 6], [2, 4, 8], [1, 1, 0], [2], id="one_split"
    ),
    pytest.param(
        [2, 2, 2], [0, 0, 1], [0, 6, 0], [4, 10, 10], [1, 1, 0], [], id="excluded"
    ),
    pytest.param(
        [3, 3, 3], [0, 0, 0], [0, 3, 6], [1, 4, 7], [1, 1, 1, 1], [], id="all_excluded"
    ),
    pytest.param(
        [3, 3, 3],
        [0, 0, 0],
        [0, 3, 6],
        [1, 4, 7],
        [0, 0, 0, 0],
        [0, 0, 3, 3],
        id="many_splits",
    ),
    pytest.param(
        [3, 3, 3, 3, 4],
        [0, 0, 0, 1, 3],
        [0, 3, 6, 0, 0],
        [1, 4, 7, 10, 10],
        [1, 1, 1, 0, 0],
        [],
        id="contiguous_over_another_child",
    ),
    pytest.param([3, 3], [0, 0], [0, 1], [1, 2], [0] * 4, [], id="touching"),
    pytest.param([3, 3], [0, 0], [0, 0], [2, 5], [0] * 4, [], id="overlapping"),
    pytest.param([3, 3], [0, 0], [0, 1.0000001], [1, 2], [0] * 4, [0, 3], id="tiny_gap"),
    pytest.param(
        [2, 2, 2], [1, 1, 0], [6, 2, 0], [8, 4, 2], [1, 1, 0], [2], id="unsorted"
    ),
    pytest.param([2], [0], [0], [1], [1, 0, 0], [], id="first_edge_at_zero"),
]


class TestSplitDisjointNodes:
    @staticmethod
    def compare(*args):
        return compare(_tsdate.split_disjoint_nodes, util.split_disjoint_nodes, *args)

    @pytest.mark.parametrize(
        ("parent", "child", "left", "right", "excluded", "expected"), SPLIT_CASES
    )
    def test_edge_cases(self, parent, child, left, right, excluded, expected):
        args = (i32(*parent), i32(*child), f64(*left), f64(*right), b1(*excluded))
        _, _, _, split_nodes = self.compare(*args).value
        assert split_nodes.tolist() == expected

    @pytest.mark.parametrize(("index", "size"), [(1, 1), (2, 1), (3, 1)])
    def test_size_mismatch(self, index, size):
        args = [i32(2, 2), i32(0, 1), f64(0, 0), f64(10, 10), b1(1, 1, 0)]
        result = self.compare(*shrink(args, index, size))
        assert result.error is ValueError

    def test_simulated(self, sim_ts):
        ts = sim_ts
        args = (
            ts.edges_parent,
            ts.edges_child,
            ts.edges_left,
            ts.edges_right,
            sample_flags(ts),
        )
        self.compare(*args)
        # also the dtypes, which `compare` ignores
        actual = _tsdate.split_disjoint_nodes(*args)
        assert all(a.dtype == np.int32 for a in actual)


def relabel_args(ts):
    """
    Arguments of `relabel_mutations_node`, with the nodes split as by `util`
    """
    parent, child, order, _ = util.split_disjoint_nodes(
        ts.edges_parent,
        ts.edges_child,
        ts.edges_left,
        ts.edges_right,
        sample_flags(ts),
    )
    return [
        ts.mutations_node,
        ts.sites_position[ts.mutations_site],
        order,
        parent,
        child,
        ts.edges_left,
        ts.edges_right,
        ts.indexes_edge_insertion_order,
        ts.indexes_edge_removal_order,
    ]


# nodes 0, 1 are samples, 2 and 3 are internal and have a gap at [4, 6)
GAP_NODES = [(1, 0), (1, 0), (0, 1), (0, 2)]
GAP_EDGES = [(0, 4, 2, 0), (0, 4, 2, 1), (6, 10, 2, 0), (6, 10, 2, 1)]
GAP_EDGES += [(0, 4, 3, 2), (6, 10, 3, 2)]
BINARY_EDGES = [(0, 10, 2, 0), (0, 10, 2, 1)]

# nodes, edges, mutations, sequence length, exception of the oracle
RELABEL_CASES = [
    pytest.param(GAP_NODES, GAP_EDGES, [], 10.0, None, id="no_mutations"),
    pytest.param(
        GAP_NODES[:3],
        BINARY_EDGES,
        [(1, 0), (2, 2), (3, 1)],
        10.0,
        None,
        id="no_splits",
    ),
    # the sweep stops at the largest edge right coordinate: later mutations are NULL
    pytest.param(
        GAP_NODES, GAP_EDGES, [(1, 2), (10.0, 2)], 20.0, None, id="beyond_last_edge"
    ),
    pytest.param(
        GAP_NODES,
        BINARY_EDGES,
        [(1, 3)],
        10.0,
        MUTATION_NODE_NOT_IN_TREE,
        id="node_not_in_tree",
    ),
]


class TestRelabelMutationsNode:
    @staticmethod
    def compare(args):
        return compare(_tsdate.relabel_mutations_node, util.relabel_mutations_node, *args)

    @pytest.mark.parametrize(
        ("nodes", "edges", "mutations", "length", "expected_error"), RELABEL_CASES
    )
    def test_edge_cases(self, nodes, edges, mutations, length, expected_error):
        ts = make_ts(nodes, edges, mutations, length)
        result = self.compare(relabel_args(ts))
        assert result.error == expected_error

    def test_mutations_on_split_nodes(self):
        mutations = [(0.5, 0), (1, 2), (2, 3), (5, 0), (7, 2), (8, 3)]
        ts = make_ts(GAP_NODES, GAP_EDGES, mutations, 10.0)
        result = self.compare(relabel_args(ts))
        # right-hand pieces of nodes 2 and 3 are the new nodes 4 and 5; the mutation
        # at 5 is in the gap, on the (unsplit) sample 0
        assert result.value.tolist() == [0, 2, 3, 0, 4, 5]

    @pytest.mark.parametrize(("index", "size"), [(4, 2), (7, 2), (1, 1)])
    def test_size_mismatch(self, index, size):
        ts = make_ts(GAP_NODES, GAP_EDGES, [(1, 2), (7, 2)], 10.0)
        result = self.compare(shrink(relabel_args(ts), index, size))
        assert result.error is ValueError

    def test_simulated(self, sim_ts):
        args = relabel_args(sim_ts)
        self.compare(args)
        # also the dtype, which `compare` ignores
        actual = _tsdate.relabel_mutations_node(*args)
        assert actual.dtype == np.int32


# times, fixed, parents, children, epsilon, max_iterations, outcome category
# node 2 (age 1) is above samples 0, 1; node 3 (age 0.5) is above node 2: violated
VIOLATED = ([0, 0, 1.0, 0.5], [1, 1, 0, 0], [2, 3], [0, 2])
CONSISTENT = ([0, 0, 1.0], [1, 1, 0], [2, 2], [0, 1])
# one edge with parent age 0 and child age 1 (a violation); node 0 is the parent
INVERTED = ([0.0, 1.0], [0, 0], [0], [1])
CONSTRAIN_EDGE_CASES = [
    pytest.param(*CONSISTENT, 1e-6, 0, "finite", id="consistent_forced"),
    pytest.param(*CONSISTENT, 1e-6, 10, "finite", id="consistent_iterated"),
    pytest.param(*CONSISTENT, 1.0, 0, "finite", id="branch_length_equals_epsilon"),
    pytest.param(*CONSISTENT, 1.0, 10, "finite", id="branch_length_equals_epsilon_it"),
    pytest.param([0, 1.0], [1, 0], [1], [0], 2.0, 0, "finite", id="shorter_than_epsilon"),
    pytest.param(*VIOLATED, 1e-6, -1, "finite", id="negative_iterations"),
    pytest.param(*INVERTED, 0.1, 100, "finite", id="free_nodes_meet"),
    pytest.param([0.0, 1.0], [1, 0], [0], [1], 0.1, 100, "finite", id="fixed_parent"),
    pytest.param([0.0, 1.0], [0, 1], [0], [1], 0.1, 100, "finite", id="fixed_child"),
    # NOTE: current behaviour, possibly a bug: two fixed nodes on a
    # violated edge are an error (reachable via `constrain_ages` with
    # inconsistent ancient sample ages); with no iterations the forcing step
    # silently changes the age of a fixed node
    pytest.param(
        [0.0, 1.0],
        [1, 1],
        [0],
        [1],
        0.1,
        100,
        FIXED_NODES_VIOLATE_CONSTRAINT,
        id="both_fixed",
    ),
    pytest.param([0.0, 1.0], [1, 1], [0], [1], 0.1, 0, "finite", id="both_fixed_forced"),
    pytest.param([0.0, 1.0], [1, 0], [], [], 0.1, 10, "finite", id="no_edges"),
    pytest.param([], [], [], [], 0.1, 10, "finite", id="no_nodes"),
    # NaN comparisons are all false: no branch is a violation, and NaN is carried
    pytest.param([0.0, NAN], [1, 0], [1], [0], 0.1, 3, "nan", id="nan_child"),
    pytest.param([NAN, 1.0], [1, 0], [1], [0], 0.1, 3, "nan", id="nan_parent"),
    pytest.param([0.0, 1.0], [1, 0], [1], [0], -0.5, 3, "finite", id="negative_epsilon"),
    pytest.param(
        [0.0, 0.0], [1, 0], [1], [0], 0.0, 3, "finite", id="equal_ages_zero_eps"
    ),
    pytest.param(
        [1.0] * 5,
        [1, 0, 0, 0, 0],
        [1, 2, 3, 4],
        [0, 1, 2, 3],
        1e-3,
        1000,
        "finite",
        id="slow_convergence_chain",
    ),
]


class TestConstrainAges:
    @staticmethod
    def args(times, fixed, parents, children, epsilon, max_iterations):
        return (
            f64(*times),
            b1(*fixed),
            i32(*parents),
            i32(*children),
            epsilon,
            max_iterations,
        )

    @staticmethod
    def compare(args):
        return compare(_tsdate.constrain_ages, util.constrain_ages, *args)

    @pytest.mark.parametrize("max_iterations", [0, 1, 1000])
    @pytest.mark.parametrize("epsilon", [0.0, 1e-6, 0.5])
    def test_violation(self, epsilon, max_iterations):
        self.compare(self.args(*VIOLATED, epsilon, max_iterations))

    @pytest.mark.parametrize(
        ("times", "fixed", "parents", "children", "epsilon", "max_iterations", "kind"),
        CONSTRAIN_EDGE_CASES,
    )
    def test_edge_cases(
        self, times, fixed, parents, children, epsilon, max_iterations, kind
    ):
        args = self.args(times, fixed, parents, children, epsilon, max_iterations)
        assert_outcome(_tsdate.constrain_ages, util.constrain_ages, args, kind)

    @pytest.mark.parametrize(
        "args",
        [
            pytest.param(([0, 1], [1], [1], [0]), id="fixed"),
            pytest.param(([0, 1], [1, 0], [1], [0, 0]), id="children"),
        ],
    )
    def test_size_mismatch(self, args):
        result = self.compare(self.args(*args, 0.1, 3))
        assert result.error is ValueError

    @pytest.mark.parametrize("mode", ["iterated", "forced"])
    @pytest.mark.parametrize("seed", range(2))
    def test_simulated(self, sim_ts, seed, mode):
        ts = sim_ts
        rng = np.random.default_rng(seed)
        fixed = sample_flags(ts)
        low, high, epsilon, max_iterations = {
            "iterated": (0.9, 1.1, 1e-6, 50),
            "forced": (0.5, 1.5, 1e-2, 0),
        }[mode]
        nodes_time = ts.nodes_time * rng.uniform(low, high, ts.num_nodes)
        nodes_time[fixed] = ts.nodes_time[fixed]
        args = (
            nodes_time,
            fixed,
            ts.edges_parent,
            ts.edges_child,
            epsilon,
            max_iterations,
        )
        self.compare(args)


# nodes, edges, expected for (samples skipped, samples not skipped)
UNARY_CASES = [
    pytest.param([(1, 0), (1, 0), (0, 1)], BINARY_EDGES, False, False, id="binary"),
    pytest.param([(1, 0), (1, 0)], [], False, False, id="no_edges"),
    pytest.param(
        [(1, 0), (1, 0), (0, 1), (0, 2)],
        BINARY_EDGES + [(0, 10, 3, 2)],
        True,
        True,
        id="unary_internal_node",
    ),
    pytest.param(
        # sample 3 has the single child 0; the root has three children
        [(1, 0), (1, 0), (1, 0), (1, 1), (0, 2)],
        [(0, 10, 3, 0), (0, 10, 4, 3), (0, 10, 4, 1), (0, 10, 4, 2)],
        False,
        True,
        id="unary_sample",
    ),
    pytest.param(
        # node 3 is unary only over [4, 6)
        [(1, 0), (1, 0), (1, 0), (0, 1), (0, 2)],
        [(0, 4, 3, 0), (0, 4, 3, 1), (4, 6, 3, 0), (6, 10, 3, 0)]
        + [(6, 10, 3, 1), (0, 10, 4, 3), (0, 10, 4, 2)],
        True,
        True,
        id="unary_in_middle_tree",
    ),
    pytest.param(
        # node 3 loses child 0 and gains child 2 at 5: it always has two children
        [(1, 0), (1, 0), (1, 0), (0, 1)],
        [(0, 5, 3, 0), (0, 10, 3, 1), (5, 10, 3, 2)],
        False,
        False,
        id="child_replaced_at_breakpoint",
    ),
    pytest.param(
        [(1, 0), (1, 0), (0, 1)],
        [(0, 5, 2, 0), (0, 5, 2, 1), (5, 10, 2, 0)],
        True,
        True,
        id="unary_in_last_tree",
    ),
    pytest.param(
        [(1, 0), (1, 0), (0, 1)],
        [(0, 5, 2, 0), (5, 10, 2, 0), (5, 10, 2, 1)],
        True,
        True,
        id="unary_in_first_tree",
    ),
]


class TestContainsUnaryNodes:
    @staticmethod
    def compare(ts, mask):
        return compare(
            _tsdate.contains_unary_nodes,
            util.contains_unary_nodes,
            mask,
            ts.edges_parent,
            ts.edges_left,
            ts.edges_right,
            ts.indexes_edge_insertion_order,
            ts.indexes_edge_removal_order,
            ts.sequence_length,
        )

    @staticmethod
    def mask(ts, skip_samples):
        mask = np.full(ts.num_nodes, False)
        if skip_samples:
            mask[ts.samples()] = True
        return mask

    @pytest.mark.parametrize(("nodes", "edges", "skipped", "not_skipped"), UNARY_CASES)
    def test_hand_built(self, nodes, edges, skipped, not_skipped):
        ts = make_ts(nodes, edges, [], 10.0)
        result = self.compare(ts, self.mask(ts, True))
        assert bool(result.value) is skipped
        result = self.compare(ts, self.mask(ts, False))
        assert bool(result.value) is not_skipped

    def test_all_nodes_masked(self):
        nodes = [(1, 0), (0, 1), (0, 2)]
        ts = make_ts(nodes, [(0, 10, 1, 0), (0, 10, 2, 1)], [], 10.0)
        result = self.compare(ts, np.full(ts.num_nodes, True))
        assert not result.value

    @pytest.mark.parametrize("skip_samples", [True, False])
    @pytest.mark.parametrize("keep", [None, 3, 20])
    def test_simulated(self, sim_ts, keep, skip_samples):
        ts = sim_ts
        if keep is not None:
            ts = ts.simplify(ts.samples()[:keep], keep_unary=True)
        self.compare(ts, self.mask(ts, skip_samples))
