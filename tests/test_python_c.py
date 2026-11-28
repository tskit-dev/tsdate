"""
Tests for the low-level _tsdate C module.
"""

import re

import _tsdate
import msprime
import numpy as np
import pytest

import tsdate

ERR_GENERIC = -1
ERR_NO_MEMORY = -2
ERR_BAD_PARAM = -3
ERR_ZERO_DIVISION = -4
ERR_UNSORTED_QUANTILES = -11
ERR_UNPHASED_INDIVIDUAL_EDGES = -16
ERR_BLOCK_OUT_OF_BOUNDS = -17
ERR_PHASE_OUT_OF_RANGE = -18
ERR_UNPHASED_LIKELIHOOD_NOT_CONSERVED = -19
ERR_MUTATION_NODE_NOT_IN_TREE = -20
ERR_FIXED_NODES_VIOLATE_CONSTRAINT = -21
ERR_NONPOSITIVE_INTERVALS = -22
ERR_ZERO_EDGE_SPAN = -24
ERR_NON_INCREASING_BREAKPOINTS = -25
ERR_INVALID_QUANTILE_WIDTH = -26
ERR_INVALID_POSTERIOR = -27
ERR_TIME_BELOW_FIRST_BREAKPOINT = -28
ERR_NO_POSTERIORS = -29
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


def sweep_kwargs(ts, parents=True):
    """
    The edge, index and mutation arrays of ``ts`` as keyword arguments
    """
    kwargs = {
        "mutations_node": ts.mutations_node,
        "mutations_position": ts.sites_position[ts.mutations_site],
        "edges_parent": ts.edges_parent,
        "edges_child": ts.edges_child,
        "edges_left": ts.edges_left,
        "edges_right": ts.edges_right,
        "indexes_insert": ts.indexes_edge_insertion_order,
        "indexes_remove": ts.indexes_edge_removal_order,
    }
    if not parents:
        del kwargs["edges_parent"]
    return kwargs


EDGE_ARRAYS = ["edges_parent", "edges_child", "edges_left", "edges_right"]
INDEXES = ["indexes_insert", "indexes_remove"]
MUTATION_ARRAYS = ["mutations_node", "mutations_position"]


def function_specs(ts):
    """
    For each function of the module that tsdate calls: valid keyword arguments, the
    groups of arrays that must have the same length, and the arrays of two columns
    """
    samples = np.zeros(ts.num_nodes, dtype=bool)
    samples[ts.samples()] = True
    unphased = np.ones(ts.num_individuals, dtype=bool)
    _, blocks_edges, mutations_block = tsdate.phasing.block_singletons(ts, unphased)
    likelihood, _ = tsdate.rescaling.count_mutations(ts)
    constraints = np.zeros((ts.num_nodes, 2))
    constraints[:, 1] = np.inf
    constraints[samples] = 0.0
    breaks = np.array([0.0, 1.0, 10.0]), np.array([0.0, 2.0, 20.0])
    posteriors = np.ones((ts.num_nodes, 2))
    return {
        "block_singletons": (
            dict(
                individuals_unphased=unphased,
                nodes_individual=ts.nodes_individual,
                **sweep_kwargs(ts, parents=False),
                sequence_length=ts.sequence_length,
            ),
            [EDGE_ARRAYS[1:] + INDEXES, MUTATION_ARRAYS],
            [],
        ),
        "reallocate_unphased": (
            dict(
                edges_likelihood=likelihood,
                mutations_phase=np.full(ts.num_mutations, 0.5),
                mutations_block=mutations_block,
                blocks_edges=blocks_edges,
            ),
            [["mutations_phase", "mutations_block"]],
            ["edges_likelihood", "blocks_edges"],
        ),
        "mutation_frequency": (
            dict(
                nodes_sample=samples[:, np.newaxis],
                **sweep_kwargs(ts),
                sequence_length=ts.sequence_length,
            ),
            [EDGE_ARRAYS + INDEXES, MUTATION_ARRAYS],
            [],
        ),
        "split_disjoint_nodes": (
            dict(
                edges_parent=ts.edges_parent,
                edges_child=ts.edges_child,
                edges_left=ts.edges_left,
                edges_right=ts.edges_right,
                nodes_excluded=samples,
            ),
            [EDGE_ARRAYS],
            [],
        ),
        "relabel_mutations_node": (
            dict(
                mutations_node=ts.mutations_node,
                mutations_position=ts.sites_position[ts.mutations_site],
                nodes_order=np.arange(ts.num_nodes, dtype=np.int32),
                **{
                    k: v
                    for k, v in sweep_kwargs(ts).items()
                    if k.startswith(("edges", "indexes"))
                },
            ),
            [EDGE_ARRAYS + INDEXES, MUTATION_ARRAYS],
            [],
        ),
        "constrain_ages": (
            dict(
                nodes_time=ts.nodes_time,
                nodes_fixed=samples,
                edges_parent=ts.edges_parent,
                edges_child=ts.edges_child,
                epsilon=1e-6,
                max_iterations=2,
            ),
            [["nodes_time", "nodes_fixed"], ["edges_parent", "edges_child"]],
            [],
        ),
        "contains_unary_nodes": (
            dict(
                nodes_mask=samples,
                edges_parent=ts.edges_parent,
                edges_left=ts.edges_left,
                edges_right=ts.edges_right,
                indexes_insert=ts.indexes_edge_insertion_order,
                indexes_remove=ts.indexes_edge_removal_order,
                sequence_length=ts.sequence_length,
            ),
            [["edges_parent", "edges_left", "edges_right"] + INDEXES],
            [],
        ),
        "conditional_coalescent_variance": (dict(num_tips=5), [], []),
        "count_mutations": (
            dict(
                nodes_sample=samples,
                **sweep_kwargs(ts),
                sequence_length=ts.sequence_length,
                size_biased=True,
            ),
            [EDGE_ARRAYS + INDEXES, MUTATION_ARRAYS],
            [],
        ),
        "mutational_area": (
            dict(
                nodes_time=ts.nodes_time,
                edges_likelihood=likelihood,
                edges_parent=ts.edges_parent,
                edges_child=ts.edges_child,
            ),
            [["edges_parent", "edges_child", "edges_likelihood"]],
            ["edges_likelihood"],
        ),
        "mutational_timescale": (
            dict(
                nodes_time=ts.nodes_time,
                edges_likelihood=likelihood,
                nodes_fixed=samples,
                edges_parent=ts.edges_parent,
                edges_child=ts.edges_child,
                max_intervals=3,
            ),
            [
                ["edges_parent", "edges_child", "edges_likelihood"],
                ["nodes_time", "nodes_fixed"],
            ],
            ["edges_likelihood"],
        ),
        "piecewise_scale_point_estimate": (
            dict(
                point_estimate=ts.nodes_time,
                point_fixed=samples,
                original_breaks=breaks[0],
                rescaled_breaks=breaks[1],
            ),
            [["point_estimate", "point_fixed"], ["original_breaks", "rescaled_breaks"]],
            [],
        ),
        "piecewise_scale_posterior": (
            dict(
                posteriors=posteriors,
                posteriors_fixed=samples,
                original_breaks=breaks[0],
                rescaled_breaks=breaks[1],
                quantile_width=0.5,
                max_shape=1000.0,
            ),
            [
                ["posteriors", "posteriors_fixed"],
                ["original_breaks", "rescaled_breaks"],
            ],
            ["posteriors"],
        ),
        "constraints_are_consistent": (
            dict(
                constraints=constraints,
                edges_parent=ts.edges_parent,
                edges_child=ts.edges_child,
            ),
            [["edges_parent", "edges_child"]],
            ["constraints"],
        ),
    }


FUNCTIONS = [
    "block_singletons",
    "reallocate_unphased",
    "mutation_frequency",
    "split_disjoint_nodes",
    "relabel_mutations_node",
    "constrain_ages",
    "contains_unary_nodes",
    "conditional_coalescent_variance",
    "count_mutations",
    "mutational_area",
    "mutational_timescale",
    "piecewise_scale_point_estimate",
    "piecewise_scale_posterior",
    "constraints_are_consistent",
]


@pytest.fixture(scope="module")
def specs(ts):
    return function_specs(ts)


def array_arguments():
    """
    (function, argument) for every array argument
    """
    specs = function_specs(make_ts_for_specs())
    return [
        (name, arg)
        for name in FUNCTIONS
        for arg, value in specs[name][0].items()
        if isinstance(value, np.ndarray)
    ]


def scalar_arguments():
    """
    (function, argument) for every numerical scalar argument (flags take any object,
    as Python's truth values)
    """
    specs = function_specs(make_ts_for_specs())
    return [
        (name, arg)
        for name in FUNCTIONS
        for arg, value in specs[name][0].items()
        if not isinstance(value, (np.ndarray, bool))
    ]


def make_ts_for_specs():
    """
    The tree sequence of the ``ts`` fixture, for parametrising tests
    """
    ts = msprime.sim_ancestry(
        2,
        sequence_length=1e4,
        recombination_rate=1e-8,
        population_size=1e4,
        random_seed=2,
    )
    return msprime.sim_mutations(ts, rate=1e-7, random_seed=2)


def bad_dtype(value):
    """
    An array of a type that cannot be cast safely to the type of ``value``
    """
    if value.dtype == np.float64:
        return value.astype(np.complex128)
    return value.astype(np.float64)


class TestArguments:
    """
    Argument handling of each function: counts, keywords, types, dimensions and lengths
    """

    @pytest.mark.parametrize("name", FUNCTIONS)
    def test_keywords(self, specs, name):
        kwargs, _, _ = specs[name]
        getattr(_tsdate, name)(**kwargs)

    @pytest.mark.parametrize("name", FUNCTIONS)
    def test_positional(self, specs, name):
        kwargs, _, _ = specs[name]
        getattr(_tsdate, name)(*kwargs.values())

    @pytest.mark.parametrize("name", FUNCTIONS)
    def test_too_few_arguments(self, specs, name):
        args = list(specs[name][0].values())
        with pytest.raises(TypeError):
            getattr(_tsdate, name)(*args[:-1])

    @pytest.mark.parametrize("name", FUNCTIONS)
    def test_too_many_arguments(self, specs, name):
        args = list(specs[name][0].values())
        with pytest.raises(TypeError):
            getattr(_tsdate, name)(*args, args[-1])

    @pytest.mark.parametrize("name", FUNCTIONS)
    def test_unknown_keyword(self, specs, name):
        kwargs = dict(specs[name][0], not_an_argument=1)
        with pytest.raises(TypeError):
            getattr(_tsdate, name)(**kwargs)

    @pytest.mark.parametrize(("name", "arg"), array_arguments())
    def test_bad_dtype(self, specs, name, arg):
        kwargs = dict(specs[name][0])
        kwargs[arg] = bad_dtype(kwargs[arg])
        with pytest.raises(TypeError):
            getattr(_tsdate, name)(**kwargs)

    @pytest.mark.parametrize(("name", "arg"), array_arguments())
    def test_bad_dimension(self, specs, name, arg):
        kwargs = dict(specs[name][0])
        value = kwargs[arg]
        kwargs[arg] = value.reshape(-1) if value.ndim == 2 else value[np.newaxis, :]
        with pytest.raises(ValueError):
            getattr(_tsdate, name)(**kwargs)

    @pytest.mark.parametrize(("name", "arg"), array_arguments())
    def test_not_an_array(self, specs, name, arg):
        kwargs = dict(specs[name][0])
        kwargs[arg] = "not an array"
        with pytest.raises((TypeError, ValueError)):
            getattr(_tsdate, name)(**kwargs)

    @pytest.mark.parametrize(("name", "arg"), scalar_arguments())
    def test_bad_scalar(self, specs, name, arg):
        kwargs = dict(specs[name][0])
        kwargs[arg] = "not a number"
        with pytest.raises(TypeError):
            getattr(_tsdate, name)(**kwargs)

    @pytest.mark.parametrize("name", FUNCTIONS)
    def test_mismatched_lengths(self, specs, name):
        kwargs, groups, _ = specs[name]
        for group in groups:
            for arg in group[1:]:
                bad = dict(kwargs)
                bad[arg] = kwargs[arg][:-1]
                with pytest.raises(ValueError):
                    getattr(_tsdate, name)(**bad)

    @pytest.mark.parametrize("name", FUNCTIONS)
    def test_two_columns(self, specs, name):
        kwargs, _, two_columns = specs[name]
        for arg in two_columns:
            bad = dict(kwargs)
            bad[arg] = np.hstack((kwargs[arg], kwargs[arg][:, :1]))
            with pytest.raises(ValueError, match="two columns"):
                getattr(_tsdate, name)(**bad)


def first_blocked(kwargs):
    """
    The first mutation of ``reallocate_unphased``'s arguments that is in a block
    """
    return np.flatnonzero(kwargs["mutations_block"] >= 0)[0]


def edit(arg, change):
    """
    An edit of the keyword arguments that sets ``arg`` to ``change(kwargs)``
    """

    def apply(kwargs):
        kwargs[arg] = change(kwargs)

    return apply


def with_entry(arg, index, value):
    """
    An edit that sets entry ``index`` of array ``arg`` to ``value``; either can be a
    function of the arguments
    """

    def change(kwargs):
        array = kwargs[arg].copy()
        entry = index(kwargs) if callable(index) else index
        array[entry] = value(kwargs) if callable(value) else value
        return array

    return edit(arg, change)


def node_not_in_tree(kwargs):
    """
    A node with no edge over the position of the first mutation
    """
    position = kwargs["mutations_position"][0]
    left, right = kwargs["edges_left"], kwargs["edges_right"]
    over = (left <= position) & (position < right)
    present = set(kwargs["edges_parent"][over]) | set(kwargs["edges_child"][over])
    return min(set(range(kwargs["nodes_order"].size)) - present)


def reversed_ages(kwargs):
    return kwargs["nodes_time"].max() - kwargs["nodes_time"]


ERROR_CASES = {
    "block_singletons_three_edges": (
        "block_singletons",
        with_entry("nodes_individual", 2, 0),
        ERR_UNPHASED_INDIVIDUAL_EDGES,
    ),
    "reallocate_block": (
        "reallocate_unphased",
        with_entry("mutations_block", first_blocked, 1000),
        ERR_BLOCK_OUT_OF_BOUNDS,
    ),
    "reallocate_edge_of_block": (
        "reallocate_unphased",
        with_entry("blocks_edges", 0, 10**6),
        ERR_BLOCK_OUT_OF_BOUNDS,
    ),
    "reallocate_phase": (
        "reallocate_unphased",
        with_entry("mutations_phase", first_blocked, 1.5),
        ERR_PHASE_OUT_OF_RANGE,
    ),
    "reallocate_not_conserved": (
        "reallocate_unphased",
        with_entry("mutations_phase", first_blocked, np.nan),
        ERR_UNPHASED_LIKELIHOOD_NOT_CONSERVED,
    ),
    "relabel_node_not_in_tree": (
        "relabel_mutations_node",
        with_entry("mutations_node", 0, node_not_in_tree),
        ERR_MUTATION_NODE_NOT_IN_TREE,
    ),
    "constrain_fixed_nodes": (
        "constrain_ages",
        lambda kwargs: kwargs.update(
            nodes_fixed=np.ones_like(kwargs["nodes_fixed"]),
            nodes_time=reversed_ages(kwargs),
        ),
        ERR_FIXED_NODES_VIOLATE_CONSTRAINT,
    ),
    "timescale_intervals": (
        "mutational_timescale",
        edit("max_intervals", lambda kwargs: 0),
        ERR_NONPOSITIVE_INTERVALS,
    ),
    "timescale_zero_span": (
        "mutational_timescale",
        edit("edges_likelihood", lambda kwargs: kwargs["edges_likelihood"] * [1, 0]),
        ERR_ZERO_EDGE_SPAN,
    ),
    "point_estimate_breaks": (
        "piecewise_scale_point_estimate",
        edit("original_breaks", lambda kwargs: kwargs["original_breaks"][::-1].copy()),
        ERR_NON_INCREASING_BREAKPOINTS,
    ),
    "posterior_breaks": (
        "piecewise_scale_posterior",
        edit("rescaled_breaks", lambda kwargs: kwargs["rescaled_breaks"][::-1].copy()),
        ERR_NON_INCREASING_BREAKPOINTS,
    ),
    "posterior_quantile_width": (
        "piecewise_scale_posterior",
        edit("quantile_width", lambda kwargs: 1.0),
        ERR_INVALID_QUANTILE_WIDTH,
    ),
    "posterior_invalid": (
        "piecewise_scale_posterior",
        with_entry("posteriors", (0, 1), 0.0),
        ERR_INVALID_POSTERIOR,
    ),
    "posterior_none": (
        "piecewise_scale_posterior",
        lambda kwargs: kwargs.update(
            posteriors=np.zeros((0, 2)), posteriors_fixed=np.zeros(0, dtype=bool)
        ),
        ERR_NO_POSTERIORS,
    ),
    "posterior_below_first_break": (
        "piecewise_scale_posterior",
        edit("original_breaks", lambda kwargs: kwargs["original_breaks"] + 50.0),
        ERR_TIME_BELOW_FIRST_BREAKPOINT,
    ),
    "posterior_zero_quantile": (
        "piecewise_scale_posterior",
        with_entry("posteriors", (0, 1), np.inf),
        ERR_ZERO_DIVISION,
    ),
    "posterior_approximation": (
        "piecewise_scale_posterior",
        with_entry("posteriors", (0, 0), np.inf),
        ERR_UNSORTED_QUANTILES,
    ),
}


class TestLibraryErrors:
    """
    Errors of the library, through each function that reports them
    """

    @pytest.mark.parametrize("case", ERROR_CASES)
    def test_error(self, specs, case):
        name, change, code = ERROR_CASES[case]
        kwargs = dict(specs[name][0])
        if name == "piecewise_scale_posterior":
            kwargs["posteriors_fixed"] = np.zeros_like(kwargs["posteriors_fixed"])
        change(kwargs)
        with raises_library_error(code):
            getattr(_tsdate, name)(**kwargs)


class TestValues:
    """
    Results that the other tests do not check
    """

    def test_reallocate_unphased_leaves_input(self):
        likelihood = np.array([[1.0, 5.0], [0.0, 5.0]])
        result = _tsdate.reallocate_unphased(
            edges_likelihood=likelihood,
            mutations_phase=np.array([0.25]),
            mutations_block=np.array([0], dtype=np.int32),
            blocks_edges=np.array([[0, 1]], dtype=np.int32),
        )
        assert result.tolist() == [[0.25, 5.0], [0.75, 5.0]]
        assert likelihood.tolist() == [[1.0, 5.0], [0.0, 5.0]]

    def test_variance_one_tip(self):
        # the result is unused by tsdate
        assert _tsdate.conditional_coalescent_variance(1).tolist() == [0.0, 0.0]

    def test_variance_negative_tips(self):
        with pytest.raises(ValueError, match="non-negative"):
            _tsdate.conditional_coalescent_variance(-1)
