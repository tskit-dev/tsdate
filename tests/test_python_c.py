"""
Tests for the low-level _tsdate C module.
"""

import _tsdate
import msprime
import numpy as np
import pytest

ERR_GENERIC = -1
ERR_NO_MEMORY = -2
ERR_BAD_PARAM = -3
ALL_ERRORS = [ERR_GENERIC, ERR_NO_MEMORY, ERR_BAD_PARAM]


class TestModule:
    def test_name(self):
        assert _tsdate.__name__ == "_tsdate"

    def test_library_error_is_exception(self):
        assert issubclass(_tsdate.LibraryError, Exception)
        assert _tsdate.LibraryError.__name__ == "LibraryError"


class TestStrerror:
    def test_success_code(self):
        assert "not an error" in _tsdate.strerror(0)

    @pytest.mark.parametrize("code", ALL_ERRORS)
    def test_known_codes(self, code):
        message = _tsdate.strerror(code)
        assert isinstance(message, str)
        assert len(message) > 0

    def test_messages_distinct(self):
        messages = {_tsdate.strerror(code) for code in [0, *ALL_ERRORS]}
        assert len(messages) == len(ALL_ERRORS) + 1

    def test_unknown_code(self):
        assert _tsdate.strerror(-12345) == "Unknown error"

    @pytest.mark.parametrize("bad_arg", ["x", None, 1.5])
    def test_bad_argument(self, bad_arg):
        with pytest.raises(TypeError):
            _tsdate.strerror(bad_arg)

    def test_no_arguments(self):
        with pytest.raises(TypeError):
            _tsdate.strerror()


class TestRaiseLibraryError:
    @pytest.mark.parametrize("code", [ERR_GENERIC, ERR_BAD_PARAM])
    def test_raises_library_error(self, code):
        with pytest.raises(_tsdate.LibraryError, match=_tsdate.strerror(code)):
            _tsdate.raise_library_error(code)

    def test_no_memory_is_memory_error(self):
        with pytest.raises(MemoryError):
            _tsdate.raise_library_error(ERR_NO_MEMORY)

    def test_unknown_code(self):
        with pytest.raises(_tsdate.LibraryError, match="Unknown error"):
            _tsdate.raise_library_error(-12345)

    def test_bad_argument(self):
        with pytest.raises(TypeError):
            _tsdate.raise_library_error("x")


@pytest.fixture(scope="module")
def ts():
    """
    A small tree sequence with recombination, mutations and two diploid individuals
    """
    ts = msprime.sim_ancestry(
        2,
        sequence_length=1e4,
        recombination_rate=1e-8,
        population_size=1e4,
        random_seed=2,
    )
    return msprime.sim_mutations(ts, rate=1e-7, random_seed=2)


def bad_ids(ids, value):
    """
    A copy of ``ids`` with the last entry replaced by ``value``
    """
    ids = ids.copy()
    ids[-1] = value
    return ids


def sweep_args(ts, **replace):
    """
    The edge arrays of ``ts`` used by a sweep over its trees, with replacements
    """
    args = {
        "edges_parent": ts.edges_parent,
        "edges_child": ts.edges_child,
        "edges_left": ts.edges_left,
        "edges_right": ts.edges_right,
        "insert": ts.indexes_edge_insertion_order,
        "remove": ts.indexes_edge_removal_order,
        "mutations_node": ts.mutations_node,
    }
    args.update(replace)
    return args


def call_sweep(name, ts, a):
    """
    Call the sweep function ``name`` with the arrays ``a``
    """
    position = ts.sites_position[ts.mutations_site]
    edges = (a["edges_parent"], a["edges_child"], a["edges_left"], a["edges_right"])
    indexes = (a["insert"], a["remove"])
    if name == "count_mutations":
        sample = np.zeros(ts.num_nodes, dtype=bool)
        sample[ts.samples()] = True
        return _tsdate.count_mutations(
            sample,
            a["mutations_node"],
            position,
            *edges,
            *indexes,
            ts.sequence_length,
            True,
        )
    if name == "mutation_frequency":
        sample = np.zeros((ts.num_nodes, 1), dtype=bool)
        sample[ts.samples(), 0] = True
        return _tsdate.mutation_frequency(
            sample, a["mutations_node"], position, *edges, *indexes, ts.sequence_length
        )
    if name == "block_singletons":
        return _tsdate.block_singletons(
            np.ones(ts.num_individuals, dtype=bool),
            ts.nodes_individual,
            a["mutations_node"],
            position,
            *edges,
            *indexes,
            ts.sequence_length,
        )
    if name == "contains_unary_nodes":
        return _tsdate.contains_unary_nodes(
            np.zeros(ts.num_nodes, dtype=bool),
            a["edges_parent"],
            a["edges_left"],
            a["edges_right"],
            *indexes,
            ts.sequence_length,
            ts.num_nodes,
        )
    assert name == "relabel_mutations_node"
    return _tsdate.relabel_mutations_node(
        a["mutations_node"],
        position,
        np.arange(ts.num_nodes, dtype=np.int32),
        *edges,
        *indexes,
    )


SWEEPS = [
    "count_mutations",
    "mutation_frequency",
    "block_singletons",
    "contains_unary_nodes",
    "relabel_mutations_node",
]


class TestBadTreeArguments:
    """
    The library's errors for ids out of bounds and for edges that cannot be swept
    """

    @pytest.mark.parametrize("name", SWEEPS)
    def test_valid(self, ts, name):
        call_sweep(name, ts, sweep_args(ts))

    @pytest.mark.parametrize("name", sorted(set(SWEEPS) - {"block_singletons"}))
    @pytest.mark.parametrize("value", [-1, 2**30])
    def test_parent_out_of_bounds(self, ts, name, value):
        a = sweep_args(ts, edges_parent=bad_ids(ts.edges_parent, value))
        with pytest.raises(_tsdate.LibraryError, match="Node id out of bounds"):
            call_sweep(name, ts, a)

    @pytest.mark.parametrize("name", sorted(set(SWEEPS) - {"contains_unary_nodes"}))
    def test_mutation_node_out_of_bounds(self, ts, name):
        a = sweep_args(ts, mutations_node=bad_ids(ts.mutations_node, ts.num_nodes))
        with pytest.raises(_tsdate.LibraryError, match="Node id out of bounds"):
            call_sweep(name, ts, a)

    @pytest.mark.parametrize("name", SWEEPS)
    def test_nan_interval(self, ts, name):
        a = sweep_args(ts, edges_right=bad_ids(ts.edges_right, np.nan))
        with pytest.raises(_tsdate.LibraryError, match="Edge interval"):
            call_sweep(name, ts, a)

    @pytest.mark.parametrize("name", sorted(set(SWEEPS) - {"relabel_mutations_node"}))
    def test_interval_past_end(self, ts, name):
        right = bad_ids(ts.edges_right, 2 * ts.sequence_length)
        with pytest.raises(_tsdate.LibraryError, match="Edge interval"):
            call_sweep(name, ts, sweep_args(ts, edges_right=right))

    @pytest.mark.parametrize("name", SWEEPS)
    def test_empty_interval(self, ts, name):
        left = ts.edges_left.copy()
        left[-1] = ts.edges_right[-1]
        with pytest.raises(_tsdate.LibraryError, match="Edge interval"):
            call_sweep(name, ts, sweep_args(ts, edges_left=left))

    @pytest.mark.parametrize("name", SWEEPS)
    def test_index_out_of_bounds(self, ts, name):
        insert = bad_ids(ts.indexes_edge_insertion_order, ts.num_edges)
        with pytest.raises(_tsdate.LibraryError, match="Edge id out of bounds"):
            call_sweep(name, ts, sweep_args(ts, insert=insert))

    @pytest.mark.parametrize("name", SWEEPS)
    def test_unsorted_index(self, ts, name):
        remove = ts.indexes_edge_removal_order[::-1].copy()
        with pytest.raises(_tsdate.LibraryError, match="Edge indexes must list"):
            call_sweep(name, ts, sweep_args(ts, remove=remove))

    @pytest.mark.parametrize("name", ["count_mutations", "mutation_frequency"])
    def test_cycle(self, ts, name):
        # every edge from node 0 to node 1 and back over the whole sequence
        num_edges = 2
        a = {
            "edges_parent": np.array([1, 0], dtype=np.int32),
            "edges_child": np.array([0, 1], dtype=np.int32),
            "edges_left": np.zeros(num_edges),
            "edges_right": np.full(num_edges, ts.sequence_length),
            "insert": np.arange(num_edges, dtype=np.int32),
            "remove": np.arange(num_edges, dtype=np.int32),
            "mutations_node": ts.mutations_node,
        }
        with pytest.raises(_tsdate.LibraryError, match="Edges form a cycle"):
            call_sweep(name, ts, a)

    def test_individual_out_of_bounds(self, ts):
        nodes_individual = bad_ids(ts.nodes_individual, ts.num_individuals)
        with pytest.raises(_tsdate.LibraryError, match="Individual id out of bounds"):
            _tsdate.block_singletons(
                np.ones(ts.num_individuals, dtype=bool),
                nodes_individual,
                ts.mutations_node,
                ts.sites_position[ts.mutations_site],
                ts.edges_parent,
                ts.edges_child,
                ts.edges_left,
                ts.edges_right,
                ts.indexes_edge_insertion_order,
                ts.indexes_edge_removal_order,
                ts.sequence_length,
            )

    @pytest.mark.parametrize("value", [-1, 2**30])
    def test_constrain_ages(self, ts, value):
        with pytest.raises(_tsdate.LibraryError, match="Node id out of bounds"):
            _tsdate.constrain_ages(
                ts.nodes_time,
                np.zeros(ts.num_nodes, dtype=bool),
                ts.edges_parent,
                bad_ids(ts.edges_child, value),
                1e-6,
                3,
            )

    def test_split_disjoint_nodes(self, ts):
        with pytest.raises(_tsdate.LibraryError, match="Node id out of bounds"):
            _tsdate.split_disjoint_nodes(
                bad_ids(ts.edges_parent, ts.num_nodes),
                ts.edges_child,
                ts.edges_left,
                ts.edges_right,
                np.zeros(ts.num_nodes, dtype=bool),
            )

    def test_relabel_nodes_order(self, ts):
        nodes_order = bad_ids(np.arange(ts.num_nodes, dtype=np.int32), ts.num_nodes)
        with pytest.raises(_tsdate.LibraryError, match="Node id out of bounds"):
            _tsdate.relabel_mutations_node(
                ts.mutations_node,
                ts.sites_position[ts.mutations_site],
                nodes_order,
                ts.edges_parent,
                ts.edges_child,
                ts.edges_left,
                ts.edges_right,
                ts.indexes_edge_insertion_order,
                ts.indexes_edge_removal_order,
            )

    def test_constraints_are_consistent(self, ts):
        constraints = np.tile([0.0, np.inf], (ts.num_nodes, 1))
        with pytest.raises(_tsdate.LibraryError, match="Node id out of bounds"):
            _tsdate.constraints_are_consistent(
                constraints, bad_ids(ts.edges_parent, -1), ts.edges_child
            )

    def test_mutational_area(self, ts):
        likelihoods = np.ones((ts.num_edges, 2))
        with pytest.raises(_tsdate.LibraryError, match="Node id out of bounds"):
            _tsdate.mutational_area(
                ts.nodes_time,
                likelihoods,
                ts.edges_parent,
                bad_ids(ts.edges_child, ts.num_nodes),
            )

    def test_mutational_timescale(self, ts):
        likelihoods = np.ones((ts.num_edges, 2))
        with pytest.raises(_tsdate.LibraryError, match="Node id out of bounds"):
            _tsdate.mutational_timescale(
                ts.nodes_time,
                likelihoods,
                np.zeros(ts.num_nodes, dtype=bool),
                ts.edges_parent,
                bad_ids(ts.edges_child, ts.num_nodes),
                2,
            )


class TestTooFew:
    """
    The library's errors for too few tips or breakpoints
    """

    def test_conditional_coalescent_variance(self):
        with pytest.raises(_tsdate.LibraryError, match="Too few tips"):
            _tsdate.conditional_coalescent_variance(0)

    def test_marginalize_over_ancestors(self):
        with pytest.raises(_tsdate.LibraryError, match="Too few tips"):
            _tsdate.marginalize_over_ancestors(np.ones((1, 2)))

    def test_piecewise_scale_point_estimate(self):
        with pytest.raises(_tsdate.LibraryError, match="No breakpoints"):
            _tsdate.piecewise_scale_point_estimate(
                np.zeros(1), np.zeros(1, dtype=bool), np.zeros(0), np.zeros(0)
            )

    def test_piecewise_scale_posterior(self):
        with pytest.raises(_tsdate.LibraryError, match="No breakpoints"):
            _tsdate.piecewise_scale_posterior(
                np.ones((1, 2)),
                np.zeros(1, dtype=bool),
                np.zeros(0),
                np.zeros(0),
                0.5,
                1000.0,
            )
