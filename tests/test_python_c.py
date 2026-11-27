"""
Tests for the low-level _tsdate C module.
"""

import re

import _tsdate
import msprime
import numpy as np
import pytest

ERR_GENERIC = -1
ERR_NO_MEMORY = -2
ERR_BAD_PARAM = -3
ERR_ZERO_DIVISION = -4
ERR_INVALID_MAX_SHAPE = -30
ERR_INVALID_MIN_STEP = -31
ERR_TWIN_EDGE_IN_PHASED_PASS = -34
ERR_NONPOSITIVE_PENALTY = -35
ALL_ERRORS = [ERR_GENERIC, ERR_NO_MEMORY, ERR_BAD_PARAM]


def raises_library_error(code):
    """
    ``pytest.raises`` for the library error ``code``
    """
    return pytest.raises(_tsdate.LibraryError, match=re.escape(_tsdate.strerror(code)))


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
            *edges[1:],
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


def ep_arrays(**replace):
    """
    Arguments of ``ExpectationPropagation``: two samples (nodes 0 and 1) below a free
    root (node 2), and a block whose nodes are both the root
    """
    args = {
        "constraints": np.array([[0.0, 0.0], [0.0, 0.0], [0.0, np.inf]]),
        "edges_parent": np.array([2, 2], dtype=np.int32),
        "edges_child": np.array([0, 1], dtype=np.int32),
        "edges_likelihood": np.array([[1.0, 1.0], [2.0, 1.0]]),
        "blocks_left": np.array([2], dtype=np.int32),
        "blocks_right": np.array([2], dtype=np.int32),
        "blocks_likelihood": np.array([[1.0, 1.0]]),
    }
    args.update(replace)
    return args


def new_ep(**replace):
    """
    An ``ExpectationPropagation`` with the arguments of ``ep_arrays``
    """
    return _tsdate.ExpectationPropagation(**ep_arrays(**replace))


EDGES = np.array([0, 1], dtype=np.int32)
BLOCKS = np.array([0], dtype=np.int32)


class TestExpectationPropagationInit:
    def test_initial_state(self):
        ep = new_ep()
        assert ep.posterior.tolist() == [[0.0, 0.0]] * 3
        assert ep.edges_lognorm.tolist() == [0.0, 0.0]
        assert ep.blocks_lognorm.tolist() == [0.0]

    def test_positional_arguments(self):
        ep = _tsdate.ExpectationPropagation(*ep_arrays().values())
        assert ep.posterior.shape == (3, 2)

    def test_no_blocks(self):
        empty = np.zeros(0, dtype=np.int32)
        ep = new_ep(
            blocks_left=empty, blocks_right=empty, blocks_likelihood=np.zeros((0, 2))
        )
        assert ep.blocks_lognorm.size == 0

    @pytest.mark.parametrize("num_args", [0, 6, 8])
    def test_wrong_number_of_arguments(self, num_args):
        args = list(ep_arrays().values()) + [None]
        with pytest.raises(TypeError):
            _tsdate.ExpectationPropagation(*args[:num_args])

    @pytest.mark.parametrize("name", list(ep_arrays()))
    def test_bad_dtype(self, name):
        value = ep_arrays()[name]
        bad = value.astype(np.float64 if value.dtype == np.int32 else np.complex128)
        with pytest.raises(TypeError):
            new_ep(**{name: bad})

    @pytest.mark.parametrize("name", list(ep_arrays()))
    def test_bad_dimension(self, name):
        value = ep_arrays()[name]
        bad = value.reshape(-1) if value.ndim == 2 else value.reshape(1, -1)
        with pytest.raises(ValueError):
            new_ep(**{name: bad})

    @pytest.mark.parametrize(
        "name", ["constraints", "edges_likelihood", "blocks_likelihood"]
    )
    def test_three_columns(self, name):
        value = ep_arrays()[name]
        bad = np.hstack((value, value[:, :1]))
        with pytest.raises(ValueError, match="two columns"):
            new_ep(**{name: bad})

    @pytest.mark.parametrize(
        ("name", "match"),
        [
            ("edges_child", "Edge arrays"),
            ("edges_likelihood", "Edge arrays"),
            ("blocks_right", "Block arrays"),
            ("blocks_likelihood", "Block arrays"),
        ],
    )
    def test_mismatched_lengths(self, name, match):
        bad = ep_arrays()[name][:-1]
        with pytest.raises(ValueError, match=match):
            new_ep(**{name: bad})

    @pytest.mark.parametrize(
        "name", ["edges_parent", "edges_child", "blocks_left", "blocks_right"]
    )
    @pytest.mark.parametrize("value", [-1, 3])
    def test_node_out_of_bounds(self, name, value):
        with pytest.raises(_tsdate.LibraryError, match="Node id out of bounds"):
            new_ep(**{name: bad_ids(ep_arrays()[name], value)})

    def test_failed_init_leaves_no_state(self):
        ep = new_ep()
        with pytest.raises(_tsdate.LibraryError):
            ep.__init__(**ep_arrays(edges_parent=np.array([2, 9], dtype=np.int32)))
        with pytest.raises(SystemError, match="not initialised"):
            ep.rescale_factors()

    def test_uninitialised(self):
        ep = _tsdate.ExpectationPropagation.__new__(_tsdate.ExpectationPropagation)
        for method in (ep.rescale_factors, ep.assemble_factors):
            with pytest.raises(SystemError, match="not initialised"):
                method()
        for name in ("posterior", "edges_lognorm", "blocks_lognorm"):
            with pytest.raises(SystemError, match="not initialised"):
                getattr(ep, name)
        with pytest.raises(SystemError, match="not initialised"):
            ep.propagate_likelihood(EDGES, False, 1000.0, 0.1)
        with pytest.raises(SystemError, match="not initialised"):
            ep.propagate_prior(np.ones(3, dtype=bool), 1000.0, 10, 1e-8)
        with pytest.raises(SystemError, match="not initialised"):
            ep.propagate_mutations(EDGES, False)

    def test_properties_are_read_only_copies(self):
        ep = new_ep()
        posterior = ep.posterior
        posterior[:] = 1.0
        assert ep.posterior.tolist() == [[0.0, 0.0]] * 3
        with pytest.raises(AttributeError):
            ep.posterior = posterior

    def test_inputs_are_copied(self):
        arrays = ep_arrays()
        ep = _tsdate.ExpectationPropagation(**arrays)
        arrays["edges_parent"][:] = 99
        ep.propagate_likelihood(EDGES, False, 1000.0, 0.1)
        assert np.all(np.isfinite(ep.posterior[2]))


class TestExpectationPropagationMethods:
    def test_iteration(self):
        ep = new_ep()
        ep.propagate_likelihood(
            order=EDGES, unphased=False, max_shape=1000.0, min_step=0.1
        )
        ep.propagate_likelihood(
            order=BLOCKS, unphased=True, max_shape=1000.0, min_step=0.1
        )
        ep.rescale_factors()
        assert np.all(ep.posterior[2] > 0)
        assert np.all(np.isfinite(ep.edges_lognorm))
        np.testing.assert_allclose(ep.assemble_factors(), ep.posterior)

    def test_propagate_prior(self):
        ep = new_ep()
        ep.propagate_likelihood(EDGES, False, 1000.0, 0.1)
        free = np.array([False, False, True])
        ep.propagate_prior(free=free, max_shape=1000.0, em_maxitt=10, em_reltol=1e-8)
        assert np.all(ep.posterior[2] > 0)

    def test_propagate_mutations(self):
        ep = new_ep()
        ep.propagate_likelihood(EDGES, False, 1000.0, 0.1)
        edges = np.array([0, -1, 1], dtype=np.int32)
        posterior, phase = ep.propagate_mutations(mutations_edge=edges, unphased=False)
        assert posterior.shape == (3, 2)
        assert np.all(np.isnan(posterior[1]))
        assert phase.tolist()[1] == 1.0
        assert np.all(np.isfinite(posterior[[0, 2]]))

    @pytest.mark.parametrize(
        "method", ["propagate_likelihood", "propagate_prior", "propagate_mutations"]
    )
    def test_wrong_number_of_arguments(self, method):
        with pytest.raises(TypeError):
            getattr(new_ep(), method)(EDGES)

    @pytest.mark.parametrize("method", ["rescale_factors", "assemble_factors"])
    def test_no_arguments(self, method):
        with pytest.raises(TypeError):
            getattr(new_ep(), method)(1)

    def test_bad_arrays(self):
        ep = new_ep()
        with pytest.raises(TypeError):
            ep.propagate_likelihood(EDGES.astype(np.float64), False, 1000.0, 0.1)
        with pytest.raises(ValueError):
            ep.propagate_likelihood(EDGES.reshape(1, -1), False, 1000.0, 0.1)
        with pytest.raises(TypeError):
            ep.propagate_prior(np.ones(3), 1000.0, 10, 1e-8)
        with pytest.raises(ValueError, match="one entry per node"):
            ep.propagate_prior(np.ones(2, dtype=bool), 1000.0, 10, 1e-8)
        with pytest.raises(TypeError):
            ep.propagate_mutations(EDGES.astype(np.float64), False)
        with pytest.raises(ValueError):
            ep.propagate_mutations(EDGES.reshape(1, -1), False)

    @pytest.mark.parametrize(
        ("order", "unphased", "match"),
        [
            ([2], False, "Edge id out of bounds"),
            ([-1], False, "Edge id out of bounds"),
            ([1], True, "Block id"),
        ],
    )
    def test_order_out_of_bounds(self, order, unphased, match):
        order = np.array(order, dtype=np.int32)
        with pytest.raises(_tsdate.LibraryError, match=match):
            new_ep().propagate_likelihood(order, unphased, 1000.0, 0.1)

    @pytest.mark.parametrize(
        ("edges", "unphased", "match"),
        [
            ([2], False, "Edge id out of bounds"),
            ([-2], False, "Edge id out of bounds"),
            ([1], True, "Block id"),
        ],
    )
    def test_mutations_edge_out_of_bounds(self, edges, unphased, match):
        edges = np.array(edges, dtype=np.int32)
        with pytest.raises(_tsdate.LibraryError, match=match):
            new_ep().propagate_mutations(edges, unphased)

    @pytest.mark.parametrize(
        ("kwargs", "code"),
        [
            (dict(max_shape=0.5), ERR_INVALID_MAX_SHAPE),
            (dict(min_step=1.0), ERR_INVALID_MIN_STEP),
        ],
    )
    def test_likelihood_parameters(self, kwargs, code):
        args = dict(order=EDGES, unphased=False, max_shape=1000.0, min_step=0.1)
        args.update(kwargs)
        with raises_library_error(code):
            new_ep().propagate_likelihood(**args)

    def test_prior_max_shape(self):
        with raises_library_error(ERR_INVALID_MAX_SHAPE):
            new_ep().propagate_prior(np.ones(3, dtype=bool), 0.5, 10, 1e-8)

    def test_twin_edge_in_phased_pass(self):
        ep = new_ep(edges_child=np.array([0, 2], dtype=np.int32))
        with raises_library_error(ERR_TWIN_EDGE_IN_PHASED_PASS):
            ep.propagate_likelihood(np.array([1], dtype=np.int32), False, 1000.0, 0.1)

    def test_nonpositive_penalty(self):
        # the fixed samples have cavities of zero rate
        with raises_library_error(ERR_NONPOSITIVE_PENALTY):
            new_ep().propagate_prior(np.ones(3, dtype=bool), 1000.0, 10, 1e-8)

    def test_zero_division(self):
        # a huge mutational target, with a fixed node of an unphased block
        ep = new_ep(
            constraints=np.array([[0.0, 0.0], [5.0, 5.0], [0.0, np.inf]]),
            blocks_right=np.array([1], dtype=np.int32),
            blocks_likelihood=np.array([[1.0, 1e300]]),
        )
        with raises_library_error(ERR_ZERO_DIVISION):
            ep.propagate_likelihood(BLOCKS, True, 1000.0, 0.1)
