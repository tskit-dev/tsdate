"""
Tests for the low-level _tsdate C module: argument handling, and each library error
through each function that reports it.
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


def with_entry(array, index, value):
    """
    A copy of ``array`` with entry ``index`` set to ``value``
    """
    array = array.copy()
    array[index] = value
    return array


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


# Two diploid individuals, with recombination (seven trees) and mutations
TS = msprime.sim_mutations(
    msprime.sim_ancestry(
        2,
        sequence_length=1e4,
        recombination_rate=1e-8,
        population_size=1e4,
        random_seed=2,
    ),
    rate=1e-7,
    random_seed=2,
)
SAMPLES = np.isin(np.arange(TS.num_nodes), TS.samples())


# Valid keyword arguments of each function, in the order of its parameters


def block_singletons_kwargs():
    return dict(
        individuals_unphased=np.ones(TS.num_individuals, dtype=bool),
        nodes_individual=TS.nodes_individual,
        mutations_node=TS.mutations_node,
        mutations_position=TS.sites_position[TS.mutations_site],
        edges_child=TS.edges_child,
        edges_left=TS.edges_left,
        edges_right=TS.edges_right,
        indexes_insert=TS.indexes_edge_insertion_order,
        indexes_remove=TS.indexes_edge_removal_order,
        sequence_length=TS.sequence_length,
    )


def reallocate_unphased_kwargs():
    # one block of edges 0 and 1, with two mutations, and a mutation in no block
    return dict(
        edges_likelihood=np.array([[2.0, 5.0], [0.0, 5.0], [1.0, 5.0]]),
        mutations_phase=np.array([0.25, 0.5, np.nan]),
        mutations_block=np.array([0, 0, -1], dtype=np.int32),
        blocks_edges=np.array([[0, 1]], dtype=np.int32),
    )


def mutation_frequency_kwargs():
    return dict(
        nodes_sample=SAMPLES[:, np.newaxis],
        mutations_node=TS.mutations_node,
        mutations_position=TS.sites_position[TS.mutations_site],
        edges_parent=TS.edges_parent,
        edges_child=TS.edges_child,
        edges_left=TS.edges_left,
        edges_right=TS.edges_right,
        indexes_insert=TS.indexes_edge_insertion_order,
        indexes_remove=TS.indexes_edge_removal_order,
        sequence_length=TS.sequence_length,
    )


def split_disjoint_nodes_kwargs():
    return dict(
        edges_parent=TS.edges_parent,
        edges_child=TS.edges_child,
        edges_left=TS.edges_left,
        edges_right=TS.edges_right,
        nodes_excluded=SAMPLES,
    )


def relabel_mutations_node_kwargs():
    return dict(
        mutations_node=TS.mutations_node,
        mutations_position=TS.sites_position[TS.mutations_site],
        nodes_order=np.arange(TS.num_nodes, dtype=np.int32),
        edges_parent=TS.edges_parent,
        edges_child=TS.edges_child,
        edges_left=TS.edges_left,
        edges_right=TS.edges_right,
        indexes_insert=TS.indexes_edge_insertion_order,
        indexes_remove=TS.indexes_edge_removal_order,
    )


def constrain_ages_kwargs():
    return dict(
        nodes_time=TS.nodes_time,
        nodes_fixed=SAMPLES,
        edges_parent=TS.edges_parent,
        edges_child=TS.edges_child,
        epsilon=1e-6,
        max_iterations=2,
    )


def contains_unary_nodes_kwargs():
    return dict(
        nodes_mask=SAMPLES,
        edges_parent=TS.edges_parent,
        edges_left=TS.edges_left,
        edges_right=TS.edges_right,
        indexes_insert=TS.indexes_edge_insertion_order,
        indexes_remove=TS.indexes_edge_removal_order,
        sequence_length=TS.sequence_length,
    )


def conditional_coalescent_variance_kwargs():
    return dict(num_tips=5)


def count_mutations_kwargs():
    return dict(
        nodes_sample=SAMPLES,
        mutations_node=TS.mutations_node,
        mutations_position=TS.sites_position[TS.mutations_site],
        edges_parent=TS.edges_parent,
        edges_child=TS.edges_child,
        edges_left=TS.edges_left,
        edges_right=TS.edges_right,
        indexes_insert=TS.indexes_edge_insertion_order,
        indexes_remove=TS.indexes_edge_removal_order,
        sequence_length=TS.sequence_length,
        size_biased=True,
    )


def mutational_area_kwargs():
    return dict(
        nodes_time=TS.nodes_time,
        edges_likelihood=np.ones((TS.num_edges, 2)),
        edges_parent=TS.edges_parent,
        edges_child=TS.edges_child,
    )


def mutational_timescale_kwargs():
    return dict(
        nodes_time=TS.nodes_time,
        edges_likelihood=np.ones((TS.num_edges, 2)),
        nodes_fixed=SAMPLES,
        edges_parent=TS.edges_parent,
        edges_child=TS.edges_child,
        max_intervals=3,
    )


def piecewise_scale_point_estimate_kwargs():
    return dict(
        point_estimate=np.array([0.0, 0.5, 5.0]),
        point_fixed=np.array([True, False, False]),
        original_breaks=np.array([0.0, 1.0, 10.0]),
        rescaled_breaks=np.array([0.0, 2.0, 20.0]),
    )


def piecewise_scale_posterior_kwargs():
    return dict(
        posteriors=np.ones((3, 2)),
        posteriors_fixed=np.array([False, False, True]),
        original_breaks=np.array([0.0, 1.0, 10.0]),
        rescaled_breaks=np.array([0.0, 2.0, 20.0]),
        quantile_width=0.5,
        max_shape=1000.0,
    )


def constraints_are_consistent_kwargs():
    constraints = np.zeros((TS.num_nodes, 2))
    constraints[~SAMPLES, 1] = np.inf
    return dict(
        constraints=constraints,
        edges_parent=TS.edges_parent,
        edges_child=TS.edges_child,
    )


def names(functions):
    """
    The names of the functions of a list of (function, make_kwargs), as test ids
    """
    return [function.__name__ for function, _ in functions]


FUNCTIONS = [
    (_tsdate.block_singletons, block_singletons_kwargs),
    (_tsdate.reallocate_unphased, reallocate_unphased_kwargs),
    (_tsdate.mutation_frequency, mutation_frequency_kwargs),
    (_tsdate.split_disjoint_nodes, split_disjoint_nodes_kwargs),
    (_tsdate.relabel_mutations_node, relabel_mutations_node_kwargs),
    (_tsdate.constrain_ages, constrain_ages_kwargs),
    (_tsdate.contains_unary_nodes, contains_unary_nodes_kwargs),
    (_tsdate.conditional_coalescent_variance, conditional_coalescent_variance_kwargs),
    (_tsdate.count_mutations, count_mutations_kwargs),
    (_tsdate.mutational_area, mutational_area_kwargs),
    (_tsdate.mutational_timescale, mutational_timescale_kwargs),
    (_tsdate.piecewise_scale_point_estimate, piecewise_scale_point_estimate_kwargs),
    (_tsdate.piecewise_scale_posterior, piecewise_scale_posterior_kwargs),
    (_tsdate.constraints_are_consistent, constraints_are_consistent_kwargs),
]


@pytest.mark.parametrize(("function", "make_kwargs"), FUNCTIONS, ids=names(FUNCTIONS))
class TestArguments:
    """
    The parsing of the arguments of each function: counts, keywords and types
    """

    def test_keywords(self, function, make_kwargs):
        function(**make_kwargs())

    def test_positional(self, function, make_kwargs):
        args = list(make_kwargs().values())
        function(*args)

    def test_too_few_arguments(self, function, make_kwargs):
        args = list(make_kwargs().values())
        with pytest.raises(TypeError):
            function(*args[:-1])

    def test_too_many_arguments(self, function, make_kwargs):
        args = list(make_kwargs().values())
        with pytest.raises(TypeError):
            function(*args, args[-1])

    def test_unknown_keyword(self, function, make_kwargs):
        kwargs = make_kwargs()
        with pytest.raises(TypeError):
            function(**kwargs, not_an_argument=1)

    def test_bad_dtype(self, function, make_kwargs):
        # float arrays as complex numbers, the others as floats: neither casts safely
        for name, value in make_kwargs().items():
            if not isinstance(value, np.ndarray):
                continue
            kwargs = make_kwargs()
            if value.dtype == np.float64:
                kwargs[name] = value.astype(np.complex128)
            else:
                kwargs[name] = value.astype(np.float64)
            with pytest.raises(TypeError):
                function(**kwargs)

    def test_bad_dimension(self, function, make_kwargs):
        for name, value in make_kwargs().items():
            if not isinstance(value, np.ndarray):
                continue
            kwargs = make_kwargs()
            if value.ndim == 2:
                kwargs[name] = value.reshape(-1)
            else:
                kwargs[name] = value.reshape(1, -1)
            with pytest.raises(ValueError):
                function(**kwargs)

    def test_not_an_array(self, function, make_kwargs):
        for name, value in make_kwargs().items():
            if not isinstance(value, np.ndarray):
                continue
            kwargs = make_kwargs()
            kwargs[name] = "not an array"
            with pytest.raises((TypeError, ValueError)):
                function(**kwargs)

    def test_not_a_number(self, function, make_kwargs):
        # flags are not included: they take any object, as its truth value
        for name, value in make_kwargs().items():
            if isinstance(value, (np.ndarray, bool)):
                continue
            kwargs = make_kwargs()
            kwargs[name] = "not a number"
            with pytest.raises(TypeError):
                function(**kwargs)


SWEEPS_WITH_SEQUENCE_LENGTH = [
    (_tsdate.block_singletons, block_singletons_kwargs),
    (_tsdate.mutation_frequency, mutation_frequency_kwargs),
    (_tsdate.contains_unary_nodes, contains_unary_nodes_kwargs),
    (_tsdate.count_mutations, count_mutations_kwargs),
]
SWEEPS = SWEEPS_WITH_SEQUENCE_LENGTH + [
    (_tsdate.relabel_mutations_node, relabel_mutations_node_kwargs),
]


class TestEdgeSweep:
    """
    The library's errors for edges that a sweep over the trees cannot traverse, through
    each function that sweeps
    """

    @pytest.mark.parametrize(("function", "make_kwargs"), SWEEPS, ids=names(SWEEPS))
    def test_nan_interval(self, function, make_kwargs):
        kwargs = make_kwargs()
        kwargs["edges_right"] = with_entry(TS.edges_right, -1, np.nan)
        with pytest.raises(_tsdate.LibraryError, match="Edge interval"):
            function(**kwargs)

    @pytest.mark.parametrize(("function", "make_kwargs"), SWEEPS, ids=names(SWEEPS))
    def test_empty_interval(self, function, make_kwargs):
        kwargs = make_kwargs()
        kwargs["edges_left"] = with_entry(TS.edges_left, -1, TS.edges_right[-1])
        with pytest.raises(_tsdate.LibraryError, match="Edge interval"):
            function(**kwargs)

    @pytest.mark.parametrize(
        ("function", "make_kwargs"),
        SWEEPS_WITH_SEQUENCE_LENGTH,
        ids=names(SWEEPS_WITH_SEQUENCE_LENGTH),
    )
    def test_interval_past_end(self, function, make_kwargs):
        kwargs = make_kwargs()
        kwargs["edges_right"] = with_entry(TS.edges_right, -1, 2 * TS.sequence_length)
        with pytest.raises(_tsdate.LibraryError, match="Edge interval"):
            function(**kwargs)

    @pytest.mark.parametrize(("function", "make_kwargs"), SWEEPS, ids=names(SWEEPS))
    def test_index_out_of_bounds(self, function, make_kwargs):
        kwargs = make_kwargs()
        insert = TS.indexes_edge_insertion_order
        kwargs["indexes_insert"] = with_entry(insert, -1, TS.num_edges)
        with pytest.raises(_tsdate.LibraryError, match="Edge id out of bounds"):
            function(**kwargs)

    @pytest.mark.parametrize(("function", "make_kwargs"), SWEEPS, ids=names(SWEEPS))
    def test_unsorted_index(self, function, make_kwargs):
        kwargs = make_kwargs()
        kwargs["indexes_remove"] = TS.indexes_edge_removal_order[::-1].copy()
        with pytest.raises(_tsdate.LibraryError, match="Edge indexes must list"):
            function(**kwargs)


class TestBlockSingletons:
    def test_mismatched_edge_arrays(self):
        for name in [
            "edges_left",
            "edges_right",
            "indexes_insert",
            "indexes_remove",
        ]:
            kwargs = block_singletons_kwargs()
            kwargs[name] = kwargs[name][:-1]
            with pytest.raises(ValueError, match="Edge arrays must have the same size"):
                _tsdate.block_singletons(**kwargs)

    def test_mismatched_mutation_arrays(self):
        kwargs = block_singletons_kwargs()
        kwargs["mutations_position"] = kwargs["mutations_position"][:-1]
        with pytest.raises(ValueError, match="Mutation arrays must have the same size"):
            _tsdate.block_singletons(**kwargs)

    def test_mutation_node_out_of_bounds(self):
        kwargs = block_singletons_kwargs()
        kwargs["mutations_node"] = with_entry(TS.mutations_node, -1, TS.num_nodes)
        with pytest.raises(_tsdate.LibraryError, match="Node id out of bounds"):
            _tsdate.block_singletons(**kwargs)

    def test_individual_out_of_bounds(self):
        kwargs = block_singletons_kwargs()
        individuals = with_entry(TS.nodes_individual, -1, TS.num_individuals)
        kwargs["nodes_individual"] = individuals
        with pytest.raises(_tsdate.LibraryError, match="Individual id out of bounds"):
            _tsdate.block_singletons(**kwargs)

    def test_individual_with_three_nodes(self):
        # nodes 0 and 1 belong to individual 0: give it node 2 as well
        assert TS.nodes_individual[:3].tolist() == [0, 0, 1]
        kwargs = block_singletons_kwargs()
        kwargs["nodes_individual"] = with_entry(TS.nodes_individual, 2, 0)
        with raises_library_error(ERR_UNPHASED_INDIVIDUAL_EDGES):
            _tsdate.block_singletons(**kwargs)


class TestReallocateUnphased:
    def test_result(self):
        # the two mutations give edge 0 their phases and edge 1 the remainders
        kwargs = reallocate_unphased_kwargs()
        result = _tsdate.reallocate_unphased(**kwargs)
        assert result.tolist() == [[0.75, 5.0], [1.25, 5.0], [1.0, 5.0]]

    def test_leaves_input(self):
        kwargs = reallocate_unphased_kwargs()
        _tsdate.reallocate_unphased(**kwargs)
        assert kwargs["edges_likelihood"].tolist() == [[2, 5], [0, 5], [1, 5]]

    def test_mismatched_mutation_arrays(self):
        kwargs = reallocate_unphased_kwargs()
        kwargs["mutations_block"] = kwargs["mutations_block"][:-1]
        with pytest.raises(ValueError, match="Mutation arrays must have the same size"):
            _tsdate.reallocate_unphased(**kwargs)

    def test_three_columns(self):
        for name in ["edges_likelihood", "blocks_edges"]:
            kwargs = reallocate_unphased_kwargs()
            value = kwargs[name]
            kwargs[name] = np.hstack((value, value[:, :1]))
            with pytest.raises(ValueError, match="must have two columns"):
                _tsdate.reallocate_unphased(**kwargs)

    def test_block_out_of_bounds(self):
        kwargs = reallocate_unphased_kwargs()
        kwargs["mutations_block"] = np.array([0, 1, -1], dtype=np.int32)
        with raises_library_error(ERR_BLOCK_OUT_OF_BOUNDS):
            _tsdate.reallocate_unphased(**kwargs)

    def test_edge_of_block_out_of_bounds(self):
        kwargs = reallocate_unphased_kwargs()
        kwargs["blocks_edges"] = np.array([[0, 3]], dtype=np.int32)
        with raises_library_error(ERR_BLOCK_OUT_OF_BOUNDS):
            _tsdate.reallocate_unphased(**kwargs)

    def test_phase_out_of_range(self):
        kwargs = reallocate_unphased_kwargs()
        kwargs["mutations_phase"] = np.array([0.25, 1.5, np.nan])
        with raises_library_error(ERR_PHASE_OUT_OF_RANGE):
            _tsdate.reallocate_unphased(**kwargs)

    def test_likelihood_not_conserved(self):
        # a NaN phase is skipped, so one mutation's likelihood is not reallocated
        kwargs = reallocate_unphased_kwargs()
        kwargs["mutations_phase"] = np.array([0.25, np.nan, np.nan])
        with raises_library_error(ERR_UNPHASED_LIKELIHOOD_NOT_CONSERVED):
            _tsdate.reallocate_unphased(**kwargs)


class TestMutationFrequency:
    def test_mismatched_edge_arrays(self):
        for name in [
            "edges_child",
            "edges_left",
            "edges_right",
            "indexes_insert",
            "indexes_remove",
        ]:
            kwargs = mutation_frequency_kwargs()
            kwargs[name] = kwargs[name][:-1]
            with pytest.raises(ValueError, match="Edge arrays must have the same size"):
                _tsdate.mutation_frequency(**kwargs)

    def test_mismatched_mutation_arrays(self):
        kwargs = mutation_frequency_kwargs()
        kwargs["mutations_position"] = kwargs["mutations_position"][:-1]
        with pytest.raises(ValueError, match="Mutation arrays must have the same size"):
            _tsdate.mutation_frequency(**kwargs)

    @pytest.mark.parametrize("value", [-1, 2**30])
    def test_parent_out_of_bounds(self, value):
        kwargs = mutation_frequency_kwargs()
        kwargs["edges_parent"] = with_entry(TS.edges_parent, -1, value)
        with pytest.raises(_tsdate.LibraryError, match="Node id out of bounds"):
            _tsdate.mutation_frequency(**kwargs)

    def test_mutation_node_out_of_bounds(self):
        kwargs = mutation_frequency_kwargs()
        kwargs["mutations_node"] = with_entry(TS.mutations_node, -1, TS.num_nodes)
        with pytest.raises(_tsdate.LibraryError, match="Node id out of bounds"):
            _tsdate.mutation_frequency(**kwargs)

    def test_cycle(self):
        # an edge from node 0 to node 1 and one back, over the whole sequence
        kwargs = mutation_frequency_kwargs()
        kwargs["edges_parent"] = np.array([1, 0], dtype=np.int32)
        kwargs["edges_child"] = np.array([0, 1], dtype=np.int32)
        kwargs["edges_left"] = np.array([0.0, 0.0])
        kwargs["edges_right"] = np.array([TS.sequence_length, TS.sequence_length])
        kwargs["indexes_insert"] = np.array([0, 1], dtype=np.int32)
        kwargs["indexes_remove"] = np.array([0, 1], dtype=np.int32)
        with pytest.raises(_tsdate.LibraryError, match="Edges form a cycle"):
            _tsdate.mutation_frequency(**kwargs)


class TestSplitDisjointNodes:
    def test_mismatched_edge_arrays(self):
        for name in ["edges_child", "edges_left", "edges_right"]:
            kwargs = split_disjoint_nodes_kwargs()
            kwargs[name] = kwargs[name][:-1]
            with pytest.raises(ValueError, match="Edge arrays must have the same size"):
                _tsdate.split_disjoint_nodes(**kwargs)

    def test_parent_out_of_bounds(self):
        kwargs = split_disjoint_nodes_kwargs()
        kwargs["edges_parent"] = with_entry(TS.edges_parent, -1, TS.num_nodes)
        with pytest.raises(_tsdate.LibraryError, match="Node id out of bounds"):
            _tsdate.split_disjoint_nodes(**kwargs)


class TestRelabelMutationsNode:
    def test_mismatched_edge_arrays(self):
        for name in [
            "edges_child",
            "edges_left",
            "edges_right",
            "indexes_insert",
            "indexes_remove",
        ]:
            kwargs = relabel_mutations_node_kwargs()
            kwargs[name] = kwargs[name][:-1]
            with pytest.raises(ValueError, match="Edge arrays must have the same size"):
                _tsdate.relabel_mutations_node(**kwargs)

    def test_mismatched_mutation_arrays(self):
        kwargs = relabel_mutations_node_kwargs()
        kwargs["mutations_position"] = kwargs["mutations_position"][:-1]
        with pytest.raises(ValueError, match="Mutation arrays must have the same size"):
            _tsdate.relabel_mutations_node(**kwargs)

    @pytest.mark.parametrize("value", [-1, 2**30])
    def test_parent_out_of_bounds(self, value):
        kwargs = relabel_mutations_node_kwargs()
        kwargs["edges_parent"] = with_entry(TS.edges_parent, -1, value)
        with pytest.raises(_tsdate.LibraryError, match="Node id out of bounds"):
            _tsdate.relabel_mutations_node(**kwargs)

    def test_mutation_node_out_of_bounds(self):
        kwargs = relabel_mutations_node_kwargs()
        kwargs["mutations_node"] = with_entry(TS.mutations_node, -1, TS.num_nodes)
        with pytest.raises(_tsdate.LibraryError, match="Node id out of bounds"):
            _tsdate.relabel_mutations_node(**kwargs)

    def test_nodes_order_out_of_bounds(self):
        kwargs = relabel_mutations_node_kwargs()
        kwargs["nodes_order"] = with_entry(kwargs["nodes_order"], -1, TS.num_nodes)
        with pytest.raises(_tsdate.LibraryError, match="Node id out of bounds"):
            _tsdate.relabel_mutations_node(**kwargs)

    def test_mutation_node_not_in_tree(self):
        position = TS.sites_position[TS.mutations_site[0]]
        absent = set(range(TS.num_nodes)) - set(TS.at(position).nodes())
        kwargs = relabel_mutations_node_kwargs()
        kwargs["mutations_node"] = with_entry(TS.mutations_node, 0, min(absent))
        with raises_library_error(ERR_MUTATION_NODE_NOT_IN_TREE):
            _tsdate.relabel_mutations_node(**kwargs)


class TestConstrainAges:
    def test_mismatched_node_arrays(self):
        kwargs = constrain_ages_kwargs()
        kwargs["nodes_fixed"] = kwargs["nodes_fixed"][:-1]
        with pytest.raises(ValueError, match="Node arrays must have the same size"):
            _tsdate.constrain_ages(**kwargs)

    def test_mismatched_edge_arrays(self):
        kwargs = constrain_ages_kwargs()
        kwargs["edges_child"] = kwargs["edges_child"][:-1]
        with pytest.raises(ValueError, match="Edge arrays must have the same size"):
            _tsdate.constrain_ages(**kwargs)

    @pytest.mark.parametrize("value", [-1, 2**30])
    def test_child_out_of_bounds(self, value):
        kwargs = constrain_ages_kwargs()
        kwargs["edges_child"] = with_entry(TS.edges_child, -1, value)
        with pytest.raises(_tsdate.LibraryError, match="Node id out of bounds"):
            _tsdate.constrain_ages(**kwargs)

    def test_fixed_nodes_violate_constraint(self):
        # every node fixed, at an age that makes every parent younger than its child
        kwargs = constrain_ages_kwargs()
        kwargs["nodes_fixed"] = np.ones(TS.num_nodes, dtype=bool)
        kwargs["nodes_time"] = TS.nodes_time.max() - TS.nodes_time
        with raises_library_error(ERR_FIXED_NODES_VIOLATE_CONSTRAINT):
            _tsdate.constrain_ages(**kwargs)


class TestContainsUnaryNodes:
    def test_mismatched_edge_arrays(self):
        for name in ["edges_left", "edges_right", "indexes_insert", "indexes_remove"]:
            kwargs = contains_unary_nodes_kwargs()
            kwargs[name] = kwargs[name][:-1]
            with pytest.raises(ValueError, match="Edge arrays must have the same size"):
                _tsdate.contains_unary_nodes(**kwargs)

    @pytest.mark.parametrize("value", [-1, 2**30])
    def test_parent_out_of_bounds(self, value):
        kwargs = contains_unary_nodes_kwargs()
        kwargs["edges_parent"] = with_entry(TS.edges_parent, -1, value)
        with pytest.raises(_tsdate.LibraryError, match="Node id out of bounds"):
            _tsdate.contains_unary_nodes(**kwargs)


class TestConditionalCoalescentVariance:
    def test_one_tip(self):
        # the result is unused by tsdate
        assert _tsdate.conditional_coalescent_variance(1).tolist() == [0.0, 0.0]

    def test_no_tips(self):
        with pytest.raises(_tsdate.LibraryError, match="Too few tips"):
            _tsdate.conditional_coalescent_variance(0)

    def test_negative_tips(self):
        with pytest.raises(ValueError, match="non-negative"):
            _tsdate.conditional_coalescent_variance(-1)


class TestCountMutations:
    def test_not_size_biased(self):
        kwargs = count_mutations_kwargs()
        kwargs["size_biased"] = False
        edges_stats, _ = _tsdate.count_mutations(**kwargs)
        assert edges_stats.shape == (TS.num_edges, 2)

    def test_mismatched_edge_arrays(self):
        for name in [
            "edges_child",
            "edges_left",
            "edges_right",
            "indexes_insert",
            "indexes_remove",
        ]:
            kwargs = count_mutations_kwargs()
            kwargs[name] = kwargs[name][:-1]
            with pytest.raises(ValueError, match="Edge arrays must have the same size"):
                _tsdate.count_mutations(**kwargs)

    def test_mismatched_mutation_arrays(self):
        kwargs = count_mutations_kwargs()
        kwargs["mutations_position"] = kwargs["mutations_position"][:-1]
        with pytest.raises(ValueError, match="Mutation arrays must have the same size"):
            _tsdate.count_mutations(**kwargs)

    @pytest.mark.parametrize("value", [-1, 2**30])
    def test_parent_out_of_bounds(self, value):
        kwargs = count_mutations_kwargs()
        kwargs["edges_parent"] = with_entry(TS.edges_parent, -1, value)
        with pytest.raises(_tsdate.LibraryError, match="Node id out of bounds"):
            _tsdate.count_mutations(**kwargs)

    def test_mutation_node_out_of_bounds(self):
        kwargs = count_mutations_kwargs()
        kwargs["mutations_node"] = with_entry(TS.mutations_node, -1, TS.num_nodes)
        with pytest.raises(_tsdate.LibraryError, match="Node id out of bounds"):
            _tsdate.count_mutations(**kwargs)

    def test_cycle(self):
        # an edge from node 0 to node 1 and one back, over the whole sequence
        kwargs = count_mutations_kwargs()
        kwargs["edges_parent"] = np.array([1, 0], dtype=np.int32)
        kwargs["edges_child"] = np.array([0, 1], dtype=np.int32)
        kwargs["edges_left"] = np.array([0.0, 0.0])
        kwargs["edges_right"] = np.array([TS.sequence_length, TS.sequence_length])
        kwargs["indexes_insert"] = np.array([0, 1], dtype=np.int32)
        kwargs["indexes_remove"] = np.array([0, 1], dtype=np.int32)
        with pytest.raises(_tsdate.LibraryError, match="Edges form a cycle"):
            _tsdate.count_mutations(**kwargs)


class TestMutationalArea:
    def test_mismatched_edge_arrays(self):
        kwargs = mutational_area_kwargs()
        kwargs["edges_child"] = kwargs["edges_child"][:-1]
        with pytest.raises(ValueError, match="Edge arrays must have the same size"):
            _tsdate.mutational_area(**kwargs)

    def test_mismatched_likelihoods(self):
        kwargs = mutational_area_kwargs()
        kwargs["edges_likelihood"] = kwargs["edges_likelihood"][:-1]
        with pytest.raises(ValueError, match="one row of two columns per edge"):
            _tsdate.mutational_area(**kwargs)

    def test_three_columns(self):
        kwargs = mutational_area_kwargs()
        kwargs["edges_likelihood"] = np.ones((TS.num_edges, 3))
        with pytest.raises(ValueError, match="one row of two columns per edge"):
            _tsdate.mutational_area(**kwargs)

    def test_child_out_of_bounds(self):
        kwargs = mutational_area_kwargs()
        kwargs["edges_child"] = with_entry(TS.edges_child, -1, TS.num_nodes)
        with pytest.raises(_tsdate.LibraryError, match="Node id out of bounds"):
            _tsdate.mutational_area(**kwargs)


class TestMutationalTimescale:
    def test_mismatched_edge_arrays(self):
        kwargs = mutational_timescale_kwargs()
        kwargs["edges_child"] = kwargs["edges_child"][:-1]
        with pytest.raises(ValueError, match="Edge arrays must have the same size"):
            _tsdate.mutational_timescale(**kwargs)

    def test_mismatched_likelihoods(self):
        kwargs = mutational_timescale_kwargs()
        kwargs["edges_likelihood"] = kwargs["edges_likelihood"][:-1]
        with pytest.raises(ValueError, match="one row of two columns per edge"):
            _tsdate.mutational_timescale(**kwargs)

    def test_three_columns(self):
        kwargs = mutational_timescale_kwargs()
        kwargs["edges_likelihood"] = np.ones((TS.num_edges, 3))
        with pytest.raises(ValueError, match="one row of two columns per edge"):
            _tsdate.mutational_timescale(**kwargs)

    def test_mismatched_node_arrays(self):
        kwargs = mutational_timescale_kwargs()
        kwargs["nodes_fixed"] = kwargs["nodes_fixed"][:-1]
        with pytest.raises(ValueError, match="nodes_fixed and nodes_time"):
            _tsdate.mutational_timescale(**kwargs)

    def test_child_out_of_bounds(self):
        kwargs = mutational_timescale_kwargs()
        kwargs["edges_child"] = with_entry(TS.edges_child, -1, TS.num_nodes)
        with pytest.raises(_tsdate.LibraryError, match="Node id out of bounds"):
            _tsdate.mutational_timescale(**kwargs)

    def test_no_intervals(self):
        kwargs = mutational_timescale_kwargs()
        kwargs["max_intervals"] = 0
        with raises_library_error(ERR_NONPOSITIVE_INTERVALS):
            _tsdate.mutational_timescale(**kwargs)

    def test_zero_edge_span(self):
        kwargs = mutational_timescale_kwargs()
        kwargs["edges_likelihood"] = np.zeros((TS.num_edges, 2))
        with raises_library_error(ERR_ZERO_EDGE_SPAN):
            _tsdate.mutational_timescale(**kwargs)


class TestPiecewiseScalePointEstimate:
    def test_mismatched_points(self):
        kwargs = piecewise_scale_point_estimate_kwargs()
        kwargs["point_fixed"] = kwargs["point_fixed"][:-1]
        with pytest.raises(ValueError, match="point_fixed and point_estimate"):
            _tsdate.piecewise_scale_point_estimate(**kwargs)

    def test_mismatched_breakpoints(self):
        kwargs = piecewise_scale_point_estimate_kwargs()
        kwargs["rescaled_breaks"] = kwargs["rescaled_breaks"][:-1]
        with pytest.raises(ValueError, match="Breakpoints must be of the same size"):
            _tsdate.piecewise_scale_point_estimate(**kwargs)

    def test_no_breakpoints(self):
        kwargs = piecewise_scale_point_estimate_kwargs()
        kwargs["original_breaks"] = np.array([])
        kwargs["rescaled_breaks"] = np.array([])
        with pytest.raises(_tsdate.LibraryError, match="No breakpoints"):
            _tsdate.piecewise_scale_point_estimate(**kwargs)

    def test_decreasing_breakpoints(self):
        kwargs = piecewise_scale_point_estimate_kwargs()
        kwargs["original_breaks"] = np.array([10.0, 1.0, 0.0])
        with raises_library_error(ERR_NON_INCREASING_BREAKPOINTS):
            _tsdate.piecewise_scale_point_estimate(**kwargs)


class TestPiecewiseScalePosterior:
    def test_mismatched_posteriors(self):
        kwargs = piecewise_scale_posterior_kwargs()
        kwargs["posteriors_fixed"] = kwargs["posteriors_fixed"][:-1]
        with pytest.raises(ValueError, match="one row per fixed flag"):
            _tsdate.piecewise_scale_posterior(**kwargs)

    def test_three_columns(self):
        kwargs = piecewise_scale_posterior_kwargs()
        kwargs["posteriors"] = np.ones((3, 3))
        with pytest.raises(ValueError, match="must have two columns"):
            _tsdate.piecewise_scale_posterior(**kwargs)

    def test_mismatched_breakpoints(self):
        kwargs = piecewise_scale_posterior_kwargs()
        kwargs["rescaled_breaks"] = kwargs["rescaled_breaks"][:-1]
        with pytest.raises(ValueError, match="Breakpoints must be of the same size"):
            _tsdate.piecewise_scale_posterior(**kwargs)

    def test_no_breakpoints(self):
        kwargs = piecewise_scale_posterior_kwargs()
        kwargs["original_breaks"] = np.array([])
        kwargs["rescaled_breaks"] = np.array([])
        with pytest.raises(_tsdate.LibraryError, match="No breakpoints"):
            _tsdate.piecewise_scale_posterior(**kwargs)

    def test_decreasing_breakpoints(self):
        kwargs = piecewise_scale_posterior_kwargs()
        kwargs["rescaled_breaks"] = np.array([20.0, 2.0, 0.0])
        with raises_library_error(ERR_NON_INCREASING_BREAKPOINTS):
            _tsdate.piecewise_scale_posterior(**kwargs)

    def test_quantile_width(self):
        kwargs = piecewise_scale_posterior_kwargs()
        kwargs["quantile_width"] = 1.0
        with raises_library_error(ERR_INVALID_QUANTILE_WIDTH):
            _tsdate.piecewise_scale_posterior(**kwargs)

    def test_invalid_posterior(self):
        kwargs = piecewise_scale_posterior_kwargs()
        kwargs["posteriors"] = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
        with raises_library_error(ERR_INVALID_POSTERIOR):
            _tsdate.piecewise_scale_posterior(**kwargs)

    def test_no_posteriors(self):
        kwargs = piecewise_scale_posterior_kwargs()
        kwargs["posteriors"] = np.zeros((0, 2))
        kwargs["posteriors_fixed"] = np.zeros(0, dtype=bool)
        with raises_library_error(ERR_NO_POSTERIORS):
            _tsdate.piecewise_scale_posterior(**kwargs)

    def test_time_below_first_breakpoint(self):
        kwargs = piecewise_scale_posterior_kwargs()
        kwargs["original_breaks"] = np.array([50.0, 51.0, 60.0])
        with raises_library_error(ERR_TIME_BELOW_FIRST_BREAKPOINT):
            _tsdate.piecewise_scale_posterior(**kwargs)

    def test_zero_quantile(self):
        kwargs = piecewise_scale_posterior_kwargs()
        kwargs["posteriors"] = np.array([[1.0, np.inf], [1.0, 1.0], [1.0, 1.0]])
        with raises_library_error(ERR_ZERO_DIVISION):
            _tsdate.piecewise_scale_posterior(**kwargs)

    def test_unsorted_quantiles(self):
        kwargs = piecewise_scale_posterior_kwargs()
        kwargs["posteriors"] = np.array([[np.inf, 1.0], [1.0, 1.0], [1.0, 1.0]])
        with raises_library_error(ERR_UNSORTED_QUANTILES):
            _tsdate.piecewise_scale_posterior(**kwargs)


class TestConstraintsAreConsistent:
    def test_mismatched_edge_arrays(self):
        kwargs = constraints_are_consistent_kwargs()
        kwargs["edges_child"] = kwargs["edges_child"][:-1]
        with pytest.raises(ValueError, match="Edge arrays must have the same size"):
            _tsdate.constraints_are_consistent(**kwargs)

    def test_three_columns(self):
        kwargs = constraints_are_consistent_kwargs()
        kwargs["constraints"] = np.zeros((TS.num_nodes, 3))
        with pytest.raises(ValueError, match="constraints must have two columns"):
            _tsdate.constraints_are_consistent(**kwargs)

    def test_parent_out_of_bounds(self):
        kwargs = constraints_are_consistent_kwargs()
        kwargs["edges_parent"] = with_entry(TS.edges_parent, -1, -1)
        with pytest.raises(_tsdate.LibraryError, match="Node id out of bounds"):
            _tsdate.constraints_are_consistent(**kwargs)


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
            new_ep(**{name: with_entry(ep_arrays()[name], -1, value)})

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

    def test_likelihood_max_shape(self):
        with raises_library_error(ERR_INVALID_MAX_SHAPE):
            new_ep().propagate_likelihood(EDGES, False, 0.5, 0.1)

    def test_likelihood_min_step(self):
        with raises_library_error(ERR_INVALID_MIN_STEP):
            new_ep().propagate_likelihood(EDGES, False, 1000.0, 1.0)

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
