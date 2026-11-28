"""
Equivalence of the tree sequence utilities of ``_tsdate`` with their oracles:
simulated tree sequences, degenerate inputs, and every error.
"""

import _tsdate
import numpy as np
import pytest
import tskit
from oracle import util
from oracle.errors import FIXED_NODES_VIOLATE_CONSTRAINT, MUTATION_NODE_NOT_IN_TREE
from oracle.helpers import assert_same_error, assert_same_result, make_ts


@pytest.fixture(params=["small_ts", "small_ts_historical"])
def sim_ts(request):
    return request.getfixturevalue(request.param)


def sample_flags(ts):
    return np.bitwise_and(ts.nodes_flags, tskit.NODE_IS_SAMPLE).astype(bool)


# edges_parent, edges_child, edges_left, edges_right, nodes_excluded, and the
# expected split_nodes (new nodes are numbered from the number of nodes, and each is
# a copy of an old node: `split_nodes` lists the old ids)
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
    @pytest.mark.parametrize(
        ("parent", "child", "left", "right", "excluded", "expected"), SPLIT_CASES
    )
    def test_edge_cases(self, parent, child, left, right, excluded, expected):
        _, _, _, split_nodes = assert_same_result(
            _tsdate.split_disjoint_nodes,
            util.split_disjoint_nodes,
            edges_parent=np.array(parent, dtype=np.int32),
            edges_child=np.array(child, dtype=np.int32),
            edges_left=np.array(left, dtype=np.float64),
            edges_right=np.array(right, dtype=np.float64),
            nodes_excluded=np.array(excluded, dtype=bool),
        )
        assert split_nodes.tolist() == expected

    def test_simulated(self, sim_ts):
        assert_same_result(
            _tsdate.split_disjoint_nodes,
            util.split_disjoint_nodes,
            edges_parent=sim_ts.edges_parent,
            edges_child=sim_ts.edges_child,
            edges_left=sim_ts.edges_left,
            edges_right=sim_ts.edges_right,
            nodes_excluded=sample_flags(sim_ts),
        )


def relabel_mutations_node_kwargs(ts):
    """
    Arguments of ``relabel_mutations_node``, with the nodes of ``ts`` split as by
    ``util.split_disjoint_nodes``
    """
    edges_parent, edges_child, nodes_order, _ = util.split_disjoint_nodes(
        edges_parent=ts.edges_parent,
        edges_child=ts.edges_child,
        edges_left=ts.edges_left,
        edges_right=ts.edges_right,
        nodes_excluded=sample_flags(ts),
    )
    return dict(
        mutations_node=ts.mutations_node,
        mutations_position=ts.sites_position[ts.mutations_site],
        nodes_order=nodes_order,
        edges_parent=edges_parent,
        edges_child=edges_child,
        edges_left=ts.edges_left,
        edges_right=ts.edges_right,
        indexes_insert=ts.indexes_edge_insertion_order,
        indexes_remove=ts.indexes_edge_removal_order,
    )


# nodes 0, 1 are samples, 2 and 3 are internal and have a gap at [4, 6)
GAP_NODES = [(1, 0), (1, 0), (0, 1), (0, 2)]
GAP_EDGES = [(0, 4, 2, 0), (0, 4, 2, 1), (6, 10, 2, 0), (6, 10, 2, 1)]
GAP_EDGES += [(0, 4, 3, 2), (6, 10, 3, 2)]
BINARY_EDGES = [(0, 10, 2, 0), (0, 10, 2, 1)]

# nodes, edges, mutations, sequence length
RELABEL_CASES = [
    pytest.param(GAP_NODES, GAP_EDGES, [], 10.0, id="no_mutations"),
    pytest.param(
        GAP_NODES[:3], BINARY_EDGES, [(1, 0), (2, 2), (3, 1)], 10.0, id="no_splits"
    ),
    # the sweep stops at the largest edge right coordinate: later mutations are NULL
    pytest.param(GAP_NODES, GAP_EDGES, [(1, 2), (10.0, 2)], 20.0, id="beyond_last_edge"),
]


class TestRelabelMutationsNode:
    @pytest.mark.parametrize(("nodes", "edges", "mutations", "length"), RELABEL_CASES)
    def test_edge_cases(self, nodes, edges, mutations, length):
        ts = make_ts(nodes, edges, mutations, length)
        kwargs = relabel_mutations_node_kwargs(ts)
        assert_same_result(
            _tsdate.relabel_mutations_node, util.relabel_mutations_node, **kwargs
        )

    def test_node_not_in_tree(self):
        # node 3 has no edges
        ts = make_ts(GAP_NODES, BINARY_EDGES, [(1, 3)], 10.0)
        kwargs = relabel_mutations_node_kwargs(ts)
        assert_same_error(
            _tsdate.relabel_mutations_node,
            util.relabel_mutations_node,
            MUTATION_NODE_NOT_IN_TREE,
            **kwargs,
        )

    def test_mutations_on_split_nodes(self):
        mutations = [(0.5, 0), (1, 2), (2, 3), (5, 0), (7, 2), (8, 3)]
        ts = make_ts(GAP_NODES, GAP_EDGES, mutations, 10.0)
        kwargs = relabel_mutations_node_kwargs(ts)
        result = assert_same_result(
            _tsdate.relabel_mutations_node, util.relabel_mutations_node, **kwargs
        )
        # right-hand pieces of nodes 2 and 3 are the new nodes 4 and 5; the mutation
        # at 5 is in the gap, on the (unsplit) sample 0
        assert result.tolist() == [0, 2, 3, 0, 4, 5]

    def test_simulated(self, sim_ts):
        kwargs = relabel_mutations_node_kwargs(sim_ts)
        assert_same_result(
            _tsdate.relabel_mutations_node, util.relabel_mutations_node, **kwargs
        )


def constrain_ages_kwargs(times, fixed, parents, children, epsilon, max_iterations):
    return dict(
        nodes_time=np.array(times, dtype=np.float64),
        nodes_fixed=np.array(fixed, dtype=bool),
        edges_parent=np.array(parents, dtype=np.int32),
        edges_child=np.array(children, dtype=np.int32),
        epsilon=epsilon,
        max_iterations=max_iterations,
    )


# nodes_time, nodes_fixed, edges_parent, edges_child
# node 2 (age 1) is above samples 0, 1; node 3 (age 0.5) is above node 2: violated
VIOLATED = ([0, 0, 1.0, 0.5], [1, 1, 0, 0], [2, 3], [0, 2])
CONSISTENT = ([0, 0, 1.0], [1, 1, 0], [2, 2], [0, 1])
# one edge with parent age 0 and child age 1 (a violation); node 0 is the parent
INVERTED = ([0.0, 1.0], [0, 0], [0], [1])

# nodes_time, nodes_fixed, edges_parent, edges_child, epsilon, max_iterations
CONSTRAIN_EDGE_CASES = [
    pytest.param(*CONSISTENT, 1e-6, 0, id="consistent_forced"),
    pytest.param(*CONSISTENT, 1e-6, 10, id="consistent_iterated"),
    pytest.param(*CONSISTENT, 1.0, 0, id="branch_length_equals_epsilon"),
    pytest.param(*CONSISTENT, 1.0, 10, id="branch_length_equals_epsilon_it"),
    pytest.param([0, 1.0], [1, 0], [1], [0], 2.0, 0, id="shorter_than_epsilon"),
    pytest.param(*VIOLATED, 1e-6, -1, id="negative_iterations"),
    pytest.param(*INVERTED, 0.1, 100, id="free_nodes_meet"),
    pytest.param([0.0, 1.0], [1, 0], [0], [1], 0.1, 100, id="fixed_parent"),
    pytest.param([0.0, 1.0], [0, 1], [0], [1], 0.1, 100, id="fixed_child"),
    # with no iterations the forcing step changes the age of a fixed node
    pytest.param([0.0, 1.0], [1, 1], [0], [1], 0.1, 0, id="both_fixed_forced"),
    pytest.param([0.0, 1.0], [1, 0], [], [], 0.1, 10, id="no_edges"),
    pytest.param([], [], [], [], 0.1, 10, id="no_nodes"),
    pytest.param([0.0, 1.0], [1, 0], [1], [0], -0.5, 3, id="negative_epsilon"),
    pytest.param([0.0, 0.0], [1, 0], [1], [0], 0.0, 3, id="equal_ages_zero_eps"),
    pytest.param(
        [1.0] * 5,
        [1, 0, 0, 0, 0],
        [1, 2, 3, 4],
        [0, 1, 2, 3],
        1e-3,
        1000,
        id="slow_convergence_chain",
    ),
]


class TestConstrainAges:
    @pytest.mark.parametrize("max_iterations", [0, 1, 1000])
    @pytest.mark.parametrize("epsilon", [0.0, 1e-6, 0.5])
    def test_violation(self, epsilon, max_iterations):
        kwargs = constrain_ages_kwargs(*VIOLATED, epsilon, max_iterations)
        assert_same_result(_tsdate.constrain_ages, util.constrain_ages, **kwargs)

    @pytest.mark.parametrize(
        ("times", "fixed", "parents", "children", "epsilon", "max_iterations"),
        CONSTRAIN_EDGE_CASES,
    )
    def test_edge_cases(self, times, fixed, parents, children, epsilon, max_iterations):
        kwargs = constrain_ages_kwargs(
            times, fixed, parents, children, epsilon, max_iterations
        )
        result = assert_same_result(_tsdate.constrain_ages, util.constrain_ages, **kwargs)
        assert np.all(np.isfinite(result))

    @pytest.mark.parametrize(
        "times", [pytest.param([0.0, np.nan]), pytest.param([np.nan, 1.0])]
    )
    def test_nan_age(self, times):
        # NaN comparisons are all false: no branch is a violation, and NaN is carried
        kwargs = constrain_ages_kwargs(times, [1, 0], [1], [0], 0.1, 3)
        result = assert_same_result(_tsdate.constrain_ages, util.constrain_ages, **kwargs)
        assert np.isnan(result).tolist() == np.isnan(times).tolist()

    def test_both_fixed(self):
        # NOTE: current behaviour, possibly a bug: two fixed nodes on a
        # violated edge are an error (reachable via `constrain_ages` with
        # inconsistent ancient sample ages)
        kwargs = constrain_ages_kwargs([0.0, 1.0], [1, 1], [0], [1], 0.1, 100)
        assert_same_error(
            _tsdate.constrain_ages,
            util.constrain_ages,
            FIXED_NODES_VIOLATE_CONSTRAINT,
            **kwargs,
        )

    @pytest.mark.parametrize("mode", ["iterated", "forced"])
    @pytest.mark.parametrize("seed", range(2))
    def test_simulated(self, sim_ts, seed, mode):
        ts = sim_ts
        rng = np.random.default_rng(seed)
        if mode == "iterated":
            noise = rng.uniform(0.9, 1.1, ts.num_nodes)
            epsilon = 1e-6
            max_iterations = 50
        else:
            noise = rng.uniform(0.5, 1.5, ts.num_nodes)
            epsilon = 1e-2
            max_iterations = 0
        fixed = sample_flags(ts)
        nodes_time = ts.nodes_time * noise
        nodes_time[fixed] = ts.nodes_time[fixed]
        assert_same_result(
            _tsdate.constrain_ages,
            util.constrain_ages,
            nodes_time=nodes_time,
            nodes_fixed=fixed,
            edges_parent=ts.edges_parent,
            edges_child=ts.edges_child,
            epsilon=epsilon,
            max_iterations=max_iterations,
        )


def contains_unary_nodes_kwargs(ts, skip_samples):
    """
    Arguments of ``contains_unary_nodes`` for ``ts``, with the samples masked if
    ``skip_samples``
    """
    nodes_mask = np.full(ts.num_nodes, False)
    if skip_samples:
        nodes_mask[ts.samples()] = True
    return dict(
        nodes_mask=nodes_mask,
        edges_parent=ts.edges_parent,
        edges_left=ts.edges_left,
        edges_right=ts.edges_right,
        indexes_insert=ts.indexes_edge_insertion_order,
        indexes_remove=ts.indexes_edge_removal_order,
        sequence_length=ts.sequence_length,
    )


# nodes, edges, and the expected results with the samples skipped and not skipped
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
    @pytest.mark.parametrize(("nodes", "edges", "skipped", "not_skipped"), UNARY_CASES)
    def test_hand_built(self, nodes, edges, skipped, not_skipped):
        ts = make_ts(nodes, edges, [], 10.0)
        kwargs = contains_unary_nodes_kwargs(ts, skip_samples=True)
        result = assert_same_result(
            _tsdate.contains_unary_nodes, util.contains_unary_nodes, **kwargs
        )
        assert result is skipped
        kwargs = contains_unary_nodes_kwargs(ts, skip_samples=False)
        result = assert_same_result(
            _tsdate.contains_unary_nodes, util.contains_unary_nodes, **kwargs
        )
        assert result is not_skipped

    def test_all_nodes_masked(self):
        nodes = [(1, 0), (0, 1), (0, 2)]
        ts = make_ts(nodes, [(0, 10, 1, 0), (0, 10, 2, 1)], [], 10.0)
        kwargs = contains_unary_nodes_kwargs(ts, skip_samples=False)
        kwargs["nodes_mask"][:] = True
        result = assert_same_result(
            _tsdate.contains_unary_nodes, util.contains_unary_nodes, **kwargs
        )
        assert result is False

    @pytest.mark.parametrize("skip_samples", [True, False])
    @pytest.mark.parametrize("keep", [None, 3, 20])
    def test_simulated(self, sim_ts, keep, skip_samples):
        ts = sim_ts
        if keep is not None:
            ts = ts.simplify(ts.samples()[:keep], keep_unary=True)
        kwargs = contains_unary_nodes_kwargs(ts, skip_samples)
        assert_same_result(
            _tsdate.contains_unary_nodes, util.contains_unary_nodes, **kwargs
        )
