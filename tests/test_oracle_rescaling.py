"""
Parity of the jitted functions in ``tsdate.rescaling`` with their oracles: random
inputs, inputs from ``ExpectationPropagation.rescale`` on simulated tree sequences,
degenerate inputs, and every ``assert``/``raise``.
"""

import numpy as np
import pytest
from oracle import rescaling
from oracle.helpers import (
    INF,
    NAN,
    assert_outcome,
    b1,
    compare,
    f64,
    i32,
    make_ts,
    shrink,
)

import tsdate.approx
from tsdate.variational import ExpectationPropagation

BREAKS_MSG = "non-increasing time breakpoints"
KLFailed = tsdate.approx.KLMinimizationFailedError


@pytest.fixture(params=["small_ts_unphased", "small_ts_historical"])
def sim_ts(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="module")
def eps(small_ts, small_ts_historical):
    """`ExpectationPropagation` after a few iterations, for each simulated ts"""
    eps = {}
    for name, ts in [
        ("small_ts", small_ts),
        ("small_ts_historical", small_ts_historical),
    ]:
        ep = ExpectationPropagation(ts, mutation_rate=1e-8)
        ep.iterate(max_shape=1000, min_step=0.1)
        eps[name] = ep
    return eps


@pytest.fixture(params=["small_ts", "small_ts_historical"])
def ep(request, eps):
    return eps[request.param]


def rescale_inputs(ep):
    """Arguments of `mutational_timescale`, as in `ExpectationPropagation.rescale`"""
    nodes_time, _ = ep.node_moments()
    nodes_fixed = ep.node_constraints[:, 0] == ep.node_constraints[:, 1]
    return (
        nodes_time,
        ep.sizebiased_likelihoods,
        nodes_fixed,
        ep.edge_parents,
        ep.edge_children,
    )


def count_args(ts, size_biased, node_is_sample=None):
    if node_is_sample is None:
        node_is_sample = np.full(ts.num_nodes, False)
        node_is_sample[ts.samples()] = True
    return [
        node_is_sample,
        ts.mutations_node,
        ts.sites_position[ts.mutations_site],
        ts.edges_parent,
        ts.edges_child,
        ts.edges_left,
        ts.edges_right,
        ts.indexes_edge_insertion_order,
        ts.indexes_edge_removal_order,
        ts.sequence_length,
        size_biased,
    ]


def piecewise_rates(rng, size, num_pieces, offset_range, rate_range):
    """Poisson counts and offsets for `size` observations, with piecewise constant rate"""
    offset = rng.uniform(*offset_range, size)
    rate = np.repeat(rng.uniform(*rate_range, num_pieces), size // num_pieces + 1)[:size]
    counts = rng.poisson(rate * offset).astype(float)
    return counts, offset


# counts, epochs, outcome category of the oracle
FIXED_CHANGEPOINT_CASES = [
    pytest.param([1.0, 2.0, 3.0], 1, "finite", id="one_epoch"),
    pytest.param([5.0], 1, "finite", id="one_epoch_one_count"),
    pytest.param([0.0, 0.0, 4.0, 0.0], 1, "finite", id="one_epoch_concentrated"),
    pytest.param([1.0, 2.0, 3.0], 10, "finite", id="more_epochs_than_counts"),
    pytest.param([0.0, 0.0, 4.0, 0.0], 4, "finite", id="concentrated_counts"),
    pytest.param([0.0, 0.0, 5.0, 5.0], 3, "finite", id="zeros_at_start"),
    pytest.param([5.0, 5.0, 0.0, 0.0], 3, "finite", id="zeros_at_end"),
    pytest.param([0.0, 5.0, 0.0, 5.0, 0.0], 3, "finite", id="zeros_in_middle"),
    pytest.param([1.0] * 4, 4, "finite", id="quantiles_exact"),
    pytest.param([1.0] * 4, 2, "finite", id="quantiles_exact_half"),
    pytest.param([1.0, 2.0], 0, AssertionError, id="zero_epochs"),
    pytest.param([1.0, 2.0], -1, AssertionError, id="negative_epochs"),
    # NOTE: current behaviour, possibly a bug: with no counts the
    # cumulative distribution is 0 / 0 = NaN, which `searchsorted` places last, so
    # every breakpoint but the last is -1
    pytest.param([0.0, 0.0, 0.0], 3, "finite", id="zero_total"),
    pytest.param([], 3, "finite", id="empty"),
]


class TestFixedChangepoints:
    @staticmethod
    def compare(impl, counts, epochs):
        return compare(
            impl.rescaling._fixed_changepoints,
            rescaling._fixed_changepoints,
            np.asarray(counts, dtype=float),
            epochs,
        )

    @pytest.mark.parametrize(
        ("counts", "epochs", "expected_kind"), FIXED_CHANGEPOINT_CASES
    )
    def test_edge_cases(self, impl, counts, epochs, expected_kind):
        assert_outcome(
            impl.rescaling._fixed_changepoints,
            rescaling._fixed_changepoints,
            (np.asarray(counts, dtype=float), epochs),
            expected_kind,
        )

    @pytest.mark.parametrize("mode", ["typical", "zeros"])
    @pytest.mark.parametrize("seed", range(3))
    def test_random(self, impl, seed, mode):
        rng = np.random.default_rng(seed)
        size = int(rng.integers(3, 300))
        counts = rng.exponential(1.0, size)
        epochs = int(rng.integers(1, 40))
        if mode == "zeros":
            counts *= rng.random(size) < 0.6
            counts[0] += 1.0
            epochs = int(rng.integers(1, 120))
        result = self.compare(impl, counts, epochs)
        # also the dtype, which `compare` ignores
        actual = impl.rescaling._fixed_changepoints(counts, epochs)
        assert actual.dtype == result.value.dtype == np.int32


COUNTS = [5.0, 6.0, 20.0, 22.0, 21.0, 3.0]
ONES = [1.0] * 6
# counts, offset, penalty, min_counts, min_offset, outcome category of the oracle
POISSON_CASES = [
    pytest.param(COUNTS, ONES, 0.0, 0.0, 0.0, "finite", id="no_penalty"),
    pytest.param(COUNTS, ONES, 2.0, 0.0, 0.0, "finite", id="penalty"),
    pytest.param(COUNTS, ONES, 1e6, 0.0, 0.0, "finite", id="huge_penalty"),
    pytest.param(COUNTS, ONES, INF, 0.0, 0.0, "finite", id="infinite_penalty"),
    # the exact counts of two observations are 21 and 22
    pytest.param(COUNTS, ONES, 2.0, 21.0, 0.0, "finite", id="min_counts_21"),
    pytest.param(COUNTS, ONES, 2.0, 22.0, 0.0, "finite", id="min_counts_22"),
    pytest.param(COUNTS, ONES, 2.0, 1000.0, 0.0, "finite", id="min_counts_unreachable"),
    pytest.param(COUNTS, ONES, 0.0, 0.0, 1.5, "finite", id="min_offset_1.5"),
    pytest.param(COUNTS, ONES, 0.0, 0.0, 6.0, "finite", id="min_offset_total"),
    pytest.param(COUNTS, ONES, 0.0, 0.0, 6.5, "finite", id="min_offset_over_total"),
    pytest.param(COUNTS, ONES, 2.0, 0.0, 100.0, "finite", id="min_offset_unreachable"),
    pytest.param(
        COUNTS, [1.0, 1.0, 0.0, 1.0, 1.0, 1.0], 2.0, 0.0, 0.0, "finite", id="zero_offset"
    ),
    pytest.param(
        COUNTS,
        [1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        2.0,
        0.0,
        0.5,
        "finite",
        id="zero_offset_min_offset",
    ),
    pytest.param([], [], 2.0, 0.0, 0.0, "finite", id="empty"),
    pytest.param([4.0], [1.0], 2.0, 0.0, 0.0, "finite", id="single_observation"),
    # log(0) is -inf and 0 * -inf is NaN, so the loss is NaN
    pytest.param([0.0] * 3, [1.0] * 3, 0.0, 0.0, 0.0, "finite", id="zero_counts"),
    pytest.param([0.0] * 3, [1.0] * 3, 2.0, 1.0, 0.0, "finite", id="zero_counts_min"),
    pytest.param(COUNTS, ONES, -1.0, 0.0, 0.0, AssertionError, id="negative_penalty"),
    pytest.param(COUNTS, ONES, NAN, 0.0, 0.0, AssertionError, id="nan_penalty"),
    pytest.param(COUNTS, ONES, 2.0, -1.0, 0.0, AssertionError, id="negative_min_counts"),
    pytest.param(COUNTS, ONES, 2.0, 0.0, -1.0, AssertionError, id="negative_min_offset"),
    pytest.param(COUNTS, ONES[:3], 2.0, 0.0, 0.0, AssertionError, id="size_mismatch"),
]


class TestPoissonChangepoints:
    @pytest.mark.parametrize(
        ("counts", "offset", "penalty", "min_counts", "min_offset", "expected_kind"),
        POISSON_CASES,
    )
    def test_edge_cases(
        self, impl, counts, offset, penalty, min_counts, min_offset, expected_kind
    ):
        args = (f64(*counts), f64(*offset), penalty, min_counts, min_offset)
        assert_outcome(
            impl.rescaling._poisson_changepoints,
            rescaling._poisson_changepoints,
            args,
            expected_kind,
        )

    @pytest.mark.parametrize("mode", ["typical", "extreme"])
    @pytest.mark.parametrize("seed", range(3))
    def test_random(self, impl, seed, mode):
        rng = np.random.default_rng(seed)
        size = int(rng.integers(8, 45))
        if mode == "typical":
            counts, offset = piecewise_rates(rng, size, 3, (0.5, 2.0), (0.5, 20.0))
            args = (counts, offset, float(rng.choice([0.0, 2.0, 10.0])), 1.0, 0.1)
        else:
            counts, offset = piecewise_rates(rng, size, 4, (0.2, 3.0), (0.2, 30.0))
            args = (counts, offset, float(rng.choice([1.0, 50.0])), 10.0, 0.5)
        result = compare(
            impl.rescaling._poisson_changepoints,
            rescaling._poisson_changepoints,
            *args,
        )
        # also the dtype, which `compare` ignores
        actual = impl.rescaling._poisson_changepoints(*args)
        assert actual.dtype == result.value.dtype == np.int32


COUNT_NODES = [(1, 0), (1, 0), (0, 1)]
COUNT_EDGES = [(0, 10, 2, 0), (0, 10, 2, 1)]
GAPPED = [(0, 4, 2, 0), (6, 10, 2, 0), (0, 10, 2, 1)]
BREAKPOINT = [(0, 5, 2, 0), (5, 10, 2, 0), (0, 10, 2, 1)]
SAMPLE_TREE_NODES = [(1, 0), (1, 0), (1, 0), (0, 1), (0, 2)]
SAMPLE_TREE_EDGES = [(0, 10, 3, 0), (0, 10, 3, 1), (0, 10, 4, 3), (0, 10, 4, 2)]

# nodes, edges, mutations, length; run with and without size bias
COUNT_CASES = [
    # node 2 is the root, so a mutation on it has no edge; several per edge
    pytest.param(
        COUNT_NODES,
        COUNT_EDGES,
        [(1, 0), (2, 0), (3, 1), (5, 2)],
        10.0,
        id="mutation_above_root",
    ),
    pytest.param(COUNT_NODES, COUNT_EDGES, [], 10.0, id="no_mutations"),
    pytest.param(COUNT_NODES, [], [(1, 0)], 10.0, id="no_edges"),
    pytest.param(COUNT_NODES, [], [], 10.0, id="no_edges_no_mutations"),
    pytest.param(
        [(1, 0), (0, 1)], [(0, 4, 1, 0)], [(1, 0), (5, 0)], 10.0, id="beyond_the_edge"
    ),
    pytest.param(COUNT_NODES, GAPPED, [(4, 0), (5, 0)], 10.0, id="in_gap"),
    pytest.param(COUNT_NODES, BREAKPOINT, [(5, 0), (4.5, 0)], 10.0, id="at_breakpoint"),
    pytest.param(
        SAMPLE_TREE_NODES,
        SAMPLE_TREE_EDGES,
        [(1, 0), (2, 3), (3, 3)],
        10.0,
        id="weighted_by_samples",
    ),
]


class TestCountMutations:
    @staticmethod
    def compare(impl, args):
        return compare(impl.rescaling._count_mutations, rescaling._count_mutations, *args)

    @pytest.mark.parametrize("size_biased", [False, True])
    @pytest.mark.parametrize(("nodes", "edges", "mutations", "length"), COUNT_CASES)
    def test_edge_cases(self, impl, nodes, edges, mutations, length, size_biased):
        ts = make_ts(nodes, edges, mutations, length)
        self.compare(impl, count_args(ts, size_biased))

    def test_mutation_edges(self, impl):
        # a mutation at a breakpoint belongs to the tree to its right; none above
        # the root; the edge id of a mutation is NULL if no edge covers it
        mutations = [(1, 0), (2, 0), (3, 1), (5, 2)]
        ts = make_ts(COUNT_NODES, COUNT_EDGES, mutations, 10.0)
        _, mutation_edge = self.compare(impl, count_args(ts, False)).value
        assert mutation_edge.tolist() == [0, 0, 1, -1]

    @pytest.mark.parametrize("size_biased", [False, True])
    def test_no_samples_flagged(self, impl, size_biased):
        ts = make_ts(COUNT_NODES, COUNT_EDGES, [(1, 0)], 10.0)
        none = np.full(ts.num_nodes, False)
        self.compare(impl, count_args(ts, size_biased, node_is_sample=none))

    def test_unsorted_mutations(self, impl):
        mutations = [(1, 0), (2, 1), (6, 0), (8, 1), (9, 2)]
        ts = make_ts(COUNT_NODES, BREAKPOINT, mutations, 10.0)
        args = count_args(ts, False)
        order = np.array([3, 0, 4, 1, 2])
        args[1], args[2] = args[1][order], args[2][order]
        self.compare(impl, args)

    @pytest.mark.parametrize(("index", "size"), [(4, 1), (7, 1), (2, 1)])
    def test_size_mismatch(self, impl, index, size):
        ts = make_ts(COUNT_NODES, COUNT_EDGES, [(1, 0), (2, 1)], 10.0)
        args = shrink(count_args(ts, False), index, size)
        result = self.compare(impl, args)
        assert result.error is AssertionError

    @pytest.mark.parametrize("size_biased", [False, True])
    def test_simulated(self, impl, sim_ts, size_biased):
        args = count_args(sim_ts, size_biased)
        result = self.compare(impl, args)
        # also the dtype, which `compare` ignores
        actual = impl.rescaling._count_mutations(*args)
        assert actual[1].dtype == result.value[1].dtype == np.int32


def area_args(nodes_time, likelihoods, parents, children):
    likelihoods = np.array(likelihoods, dtype=float)
    if likelihoods.size == 0:
        likelihoods = likelihoods.reshape(0, 2)
    return (
        f64(*nodes_time),
        likelihoods,
        i32(*parents),
        i32(*children),
    )


LIK = [[2, 10], [1, 10]]
# nodes_time, likelihoods, parents, children, exception of the oracle
AREA_CASES = [
    pytest.param([0, 0, 1], LIK, [2, 2], [0, 1], None, id="single_epoch"),
    # all nodes at time zero: no epoch, and every edge has zero length
    pytest.param([0, 0, 0], LIK, [2, 2], [0, 1], None, id="no_epochs"),
    pytest.param([0, 1], [], [], [], None, id="no_edges"),
    pytest.param([0.0], [], [], [], None, id="single_node"),
    pytest.param([], [], [], [], None, id="no_nodes"),
    pytest.param(
        [0, 0, 1, 1, 3],
        [[2, 10], [1, 10], [4, 5]],
        [2, 2, 4],
        [0, 1, 3],
        None,
        id="zero_length_edge",
    ),
    # an edge whose parent is younger than its child is skipped entirely
    pytest.param([0, 1, 0.5], LIK, [1, 2], [0, 1], None, id="negative_length_edge"),
    pytest.param([0, 0, 1, 3], [[6, 12], [3, 12]], [3, 2], [0, 1], None, id="two_epochs"),
    # NOTE: current behaviour, possibly a bug: the first epoch always
    # starts at zero, even if the youngest node is older
    pytest.param([1, 2], [[2, 10]], [1], [0], None, id="no_node_at_time_zero"),
    pytest.param(
        [3, 0, 1, 0, 1, 5],
        [[2, 10], [1, 10], [3, 10], [2, 10]],
        [4, 2, 5, 2],
        [1, 3, 0, 1],
        None,
        id="ties_and_unsorted_times",
    ),
    pytest.param([0, 0, 1], LIK, [2], [0, 1], AssertionError, id="parents_mismatch"),
    pytest.param(
        [0, 0, 1], [[1, 1]] * 3, [2, 2], [0, 1], AssertionError, id="likelihood_rows"
    ),
    pytest.param(
        [0, 0, 1],
        [[1, 1, 1]] * 2,
        [2, 2],
        [0, 1],
        AssertionError,
        id="likelihood_columns",
    ),
]


class TestMutationalArea:
    @staticmethod
    def compare(impl, *args):
        return compare(impl.rescaling.mutational_area, rescaling.mutational_area, *args)

    @pytest.mark.parametrize(
        ("nodes_time", "likelihoods", "parents", "children", "expected_error"), AREA_CASES
    )
    def test_edge_cases(
        self, impl, nodes_time, likelihoods, parents, children, expected_error
    ):
        args = area_args(nodes_time, likelihoods, parents, children)
        result = self.compare(impl, *args)
        assert result.error is expected_error

    @pytest.mark.parametrize("seed", range(2))
    def test_random(self, impl, sim_ts, seed):
        ts = sim_ts
        rng = np.random.default_rng(seed)
        times = ts.nodes_time * rng.uniform(0.8, 1.2, ts.num_nodes)
        times[ts.samples()] = ts.nodes_time[ts.samples()]
        likelihoods = rng.uniform(0, 5, (ts.num_edges, 2))
        args = (times, likelihoods, ts.edges_parent, ts.edges_child)
        result = self.compare(impl, *args)
        # also the dtype, which `compare` ignores
        actual = impl.rescaling.mutational_area(*args)
        assert actual[3].dtype == result.value[3].dtype == np.int32

    def test_simulated(self, impl, ep):
        nodes_time, likelihoods, _, parents, children = rescale_inputs(ep)
        self.compare(impl, nodes_time, likelihoods, parents, children)


def timescale_args(nodes_time, likelihoods, nodes_fixed, parents, children, k):
    return (
        f64(*nodes_time),
        np.array(likelihoods, dtype=float).reshape(-1, 2),
        b1(*nodes_fixed),
        i32(*parents),
        i32(*children),
        k,
    )


# nodes 0, 1 are samples at time 0, 2 and 3 are internal
CHAIN = (
    [0, 0, 1.0, 2.0],
    [[2, 10], [1, 10], [3, 10]],
    [1, 1, 0, 0],
    [2, 2, 3],
    [0, 1, 2],
)
# sample 1 is historical
HISTORICAL = ([0, 500, 100, 700], [[2, 10], [1, 10], [3, 10]])
HISTORICAL_EDGES = ([2, 3, 3], [0, 2, 1])


def chain(k, fixed=None):
    times, lik, nodes_fixed, parents, children = CHAIN
    return timescale_args(times, lik, fixed or nodes_fixed, parents, children, k)


def historical(k, fixed):
    times, lik = HISTORICAL
    parents, children = HISTORICAL_EDGES
    return timescale_args(times, lik, fixed, parents, children, k)


# arguments, outcome category (or exception) of the oracle
TIMESCALE_CASES = [
    pytest.param(chain(1), "finite", id="one_interval"),
    pytest.param(chain(2), "finite", id="two_intervals"),
    pytest.param(chain(3), "finite", id="as_many_as_epochs"),
    pytest.param(chain(10000), "finite", id="more_than_epochs"),
    pytest.param(chain(0), AssertionError, id="zero_intervals"),
    pytest.param(chain(-1), AssertionError, id="negative_intervals"),
    # NOTE: current behaviour, possibly a bug: `nodes_fixed` is only
    # size-checked, so which nodes are fixed does not matter
    pytest.param(chain(3, [0, 0, 0, 0]), "finite", id="no_fixed_nodes"),
    pytest.param(chain(3, [1, 1, 1, 1]), "finite", id="all_fixed_nodes"),
    pytest.param(historical(1, [1, 0, 0, 0]), "finite", id="contemporary_one"),
    pytest.param(historical(3, [1, 1, 0, 0]), "finite", id="historical_three"),
    pytest.param(historical(100, [1, 1, 0, 0]), "finite", id="historical_many"),
    # the top edge has no span, so the last epoch has no weight and is merged
    pytest.param(
        timescale_args(
            [0, 0, 1.0, 2.0],
            [[2, 10], [1, 10], [3, 0]],
            [1, 1, 0, 0],
            [2, 2, 3],
            [0, 1, 2],
            1,
        ),
        "finite",
        id="zero_span_top_edge",
    ),
    # edge 2 -> 3 is negative (3 is older than 2): skipped from the totals
    pytest.param(
        timescale_args(
            [0, 0, 2.0, 1.0],
            [[2, 10], [1, 10], [5, 10]],
            [1, 1, 0, 0],
            [2, 2, 3],
            [0, 1, 2],
            2,
        ),
        "finite",
        id="negative_length_edge",
    ),
    # NOTE: current behaviour, possibly a bug: every edge over an
    # interval has zero span (a zero mutation rate) or there are no edges
    pytest.param(
        timescale_args([0, 0, 1.0], [[2, 0], [1, 0]], [1, 1, 0], [2, 2], [0, 1], 1),
        AssertionError,
        id="zero_edge_span",
    ),
    pytest.param(
        timescale_args([0.0, 1.0], [], [1, 1], [], [], 1), AssertionError, id="no_edges"
    ),
    pytest.param(
        timescale_args([0.0, 0.0, 0.0], [[1, 10], [1, 10]], [1, 1, 1], [2, 2], [0, 1], 1),
        AssertionError,
        id="all_nodes_at_time_zero",
    ),
]


class TestMutationalTimescale:
    @staticmethod
    def compare(impl, *args):
        return compare(
            impl.rescaling.mutational_timescale, rescaling.mutational_timescale, *args
        )

    @pytest.mark.parametrize(("args", "expected"), TIMESCALE_CASES)
    def test_edge_cases(self, impl, args, expected):
        assert_outcome(
            impl.rescaling.mutational_timescale,
            rescaling.mutational_timescale,
            args,
            expected,
        )

    def test_zero_edge_span_message(self, impl):
        # the message is compared as well as the type
        args = timescale_args([0, 0, 1.0], [[2, 0], [1, 0]], [1, 1, 0], [2, 2], [0, 1], 1)
        result = self.compare(impl, *args)
        assert "Zero edge span in interval" in str(result.exception)

    @pytest.mark.parametrize("which", ["fixed", "parents", "likelihood_rows", "columns"])
    def test_size_mismatch(self, impl, which):
        args = list(chain(2))
        if which == "fixed":
            args[2] = b1(1, 1)
        elif which == "parents":
            args[3] = i32(2, 2)
        elif which == "likelihood_rows":
            args[1] = np.ones((2, 2))
        else:
            args[1] = np.ones((3, 3))
        result = self.compare(impl, *args)
        assert result.error is AssertionError

    @pytest.mark.parametrize("max_intervals", [3, 1000])
    @pytest.mark.parametrize("seed", range(2))
    def test_random(self, impl, sim_ts, seed, max_intervals):
        ts = sim_ts
        rng = np.random.default_rng(seed)
        times = ts.nodes_time * rng.uniform(0.9, 1.1, ts.num_nodes)
        times[ts.samples()] = ts.nodes_time[ts.samples()]
        likelihoods = np.column_stack(
            (
                rng.poisson(2.0, ts.num_edges).astype(float),
                rng.uniform(1, 5, ts.num_edges),
            )
        )
        fixed = np.full(ts.num_nodes, False)
        fixed[ts.samples()] = True
        args = (times, likelihoods, fixed, ts.edges_parent, ts.edges_child, max_intervals)
        self.compare(impl, *args)

    @pytest.mark.parametrize("max_intervals", [1, 10, 1000])
    def test_simulated(self, impl, ep, max_intervals):
        self.compare(impl, *rescale_inputs(ep), max_intervals)


ORIGINAL = [0.0, 1.0, 3.0]
RESCALED = [0.0, 2.0, 3.5]
# estimate, fixed, original, rescaled, outcome category of the oracle
POINT_ESTIMATE_CASES = [
    pytest.param(
        [0.0, 0.5, 1.0, 2.0, 3.0], None, ORIGINAL, RESCALED, "finite", id="inside"
    ),
    # the last scaling is zero, so everything beyond collapses to the last point
    pytest.param(
        [3.0, 4.0, 10.0, 1e300], None, ORIGINAL, RESCALED, "finite", id="beyond"
    ),
    pytest.param([INF], None, ORIGINAL, RESCALED, "nan", id="infinite"),
    # NOTE: current behaviour, possibly a bug: a time below the first
    # break has search index -1, which wraps to the last breakpoint; -inf gives NaN
    pytest.param(
        [-1.0, -1e-9, 0.0], None, ORIGINAL, RESCALED, "finite", id="below_first"
    ),
    pytest.param([-INF], None, ORIGINAL, RESCALED, "nan", id="minus_infinity"),
    pytest.param([NAN], None, ORIGINAL, RESCALED, "nan", id="nan"),
    pytest.param(
        [0.0, 0.5, 3.0, 4.0], [1, 0, 1, 1], ORIGINAL, RESCALED, "finite", id="some_fixed"
    ),
    pytest.param([NAN, 1.0], [1, 0], ORIGINAL, RESCALED, "nan", id="fixed_nan"),
    pytest.param([], None, ORIGINAL, RESCALED, "finite", id="empty"),
    pytest.param([0.0, 0.5, 7.0], None, [0.0], [0.0], "finite", id="single_breakpoint"),
    pytest.param([0.0, 0.5, 7.0], None, [0.0], [2.0], "finite", id="single_scaled"),
    pytest.param([0.0, 0.3, 1.0], None, [0.0, 5.0], [0.0, 5.0], "finite", id="identity"),
    pytest.param(
        [0.5], None, [0.0, 1.0, 3.0], [0.0, 2.0, 2.0], ValueError, id="flat_rescaled"
    ),
    pytest.param(
        [0.5], None, [0.0, 1.0, 3.0], [0.0, 2.0, 1.0], ValueError, id="decreasing"
    ),
    pytest.param(
        [0.5], None, [0.0, 1.0, 1.0], [0.0, 2.0, 3.0], ValueError, id="flat_original"
    ),
    pytest.param(
        [0.5], None, [0.0, 2.0, 1.0], [0.0, 1.0, 2.0], ValueError, id="decr_original"
    ),
    pytest.param(
        [0.5], None, [0.0, 1.0, NAN], [0.0, 1.0, 2.0], ValueError, id="nan_original"
    ),
    pytest.param(
        [0.5], None, [0.0, 1.0, 2.0], [0.0, 1.0, NAN], ValueError, id="nan_rescaled"
    ),
]


class TestPiecewiseScalePointEstimate:
    @pytest.mark.parametrize(
        ("estimate", "fixed", "original", "rescaled", "expected_kind"),
        POINT_ESTIMATE_CASES,
    )
    def test_edge_cases(self, impl, estimate, fixed, original, rescaled, expected_kind):
        estimate = f64(*estimate)
        fixed = np.full(estimate.size, False) if fixed is None else b1(*fixed)
        args = (estimate, fixed, f64(*original), f64(*rescaled))
        assert_outcome(
            impl.rescaling.piecewise_scale_point_estimate,
            rescaling.piecewise_scale_point_estimate,
            args,
            expected_kind,
        )

    def test_breaks_message(self, impl):
        # the message names the offending option, and is compared as well as the type
        args = (f64(0.5), b1(0), f64(0.0, 1.0, 1.0), f64(0.0, 2.0, 3.0))
        result = compare(
            impl.rescaling.piecewise_scale_point_estimate,
            rescaling.piecewise_scale_point_estimate,
            *args,
        )
        assert BREAKS_MSG in str(result.exception)

    @pytest.mark.parametrize("seed", range(3))
    def test_random(self, impl, seed):
        rng = np.random.default_rng(seed)
        k = int(rng.integers(2, 12))
        original = np.concatenate(([0.0], np.cumsum(rng.uniform(0.1, 5.0, k))))
        rescaled = np.concatenate(([0.0], np.cumsum(rng.uniform(0.1, 5.0, k))))
        estimate = rng.uniform(0, original[-1] * 1.2, 40)
        fixed = rng.random(40) < 0.2
        compare(
            impl.rescaling.piecewise_scale_point_estimate,
            rescaling.piecewise_scale_point_estimate,
            estimate,
            fixed,
            original,
            rescaled,
        )

    def test_simulated(self, impl, ep):
        nodes_time, likelihoods, nodes_fixed, parents, children = rescale_inputs(ep)
        original, rescaled = rescaling.mutational_timescale(
            nodes_time, likelihoods, nodes_fixed, parents, children, 10
        )
        compare(
            impl.rescaling.piecewise_scale_point_estimate,
            rescaling.piecewise_scale_point_estimate,
            nodes_time,
            nodes_fixed,
            original,
            rescaled,
        )


POSTERIORS = [[1.0, 1.0], [2.0, 3.0], [0.0, 1.0], [5.0, 2.0]]


# posteriors, keyword arguments of `posterior_args`, outcome category (or exception)
# of the oracle
POSTERIOR_CASES = [
    pytest.param(POSTERIORS, {}, "finite", id="default"),
    pytest.param(POSTERIORS, {"quantile_width": 1e-3}, "finite", id="narrow_quantile"),
    pytest.param(POSTERIORS, {"quantile_width": 0.999}, "finite", id="wide_quantile"),
    pytest.param(
        POSTERIORS, {"quantile_width": 1 - 1e-9}, "finite", id="widest_quantile"
    ),
    *[
        pytest.param(POSTERIORS, {"quantile_width": w}, AssertionError, id=f"width_{w}")
        for w in [0.0, 1.0, -0.1, 1.5, NAN, INF]
    ],
    pytest.param(POSTERIORS, {"fixed": [1, 0, 1, 0]}, "nan", id="some_fixed"),
    pytest.param(POSTERIORS, {"fixed": [1] * 4}, "nan", id="all_fixed"),
    # fixed rows are not validated
    pytest.param(
        [[-5.0, -5.0], [1.0, 1.0]], {"fixed": [1, 0]}, "nan", id="fixed_negative"
    ),
    pytest.param([[NAN, NAN], [1.0, 1.0]], {"fixed": [1, 0]}, "nan", id="fixed_nan"),
    # NOTE: current behaviour, possibly a bug: no posteriors raises
    # numpy's "zero-size array" ValueError from a debug assertion
    pytest.param(np.zeros((0, 2)), {"fixed": []}, ValueError, id="no_posteriors"),
    *[
        pytest.param([row], {}, "finite", id=f"valid_{i}")
        for i, row in enumerate(
            [[-0.99, 1.0], [0.0, 1.0], [1.0, 1e-300], [1.0, 1e300], [1e6, 1e6]]
        )
    ],
    # NOTE: current behaviour, possibly a bug: for a shape this close
    # to -1 the lower quantile underflows to zero and numba raises ZeroDivisionError
    # (IEEE would give inf), rather than KLMinimizationFailedError
    *[
        pytest.param([row], {}, ZeroDivisionError, id=f"shape_near_minus_one_{i}")
        for i, row in enumerate([[-0.9999, 1.0], [-1 + 1e-12, 1.0], [-0.9999999, 2.0]])
    ],
    pytest.param([[1.0, INF]], {}, ZeroDivisionError, id="infinite_rate"),
    # NOTE: current behaviour, possibly a bug: the error leaks out of
    # `approximate_gamma_iqr`
    pytest.param([[INF, 1.0]], {}, KLFailed, id="infinite_shape"),
    *[
        pytest.param([row], {}, AssertionError, id=f"invalid_{i}")
        for i, row in enumerate(
            [[-1.0, 1.0], [1.0, 0.0], [1.0, -1.0], [NAN, 1.0], [1.0, NAN], [-INF, 1.0]]
        )
    ],
    pytest.param(
        [[1.0, 1.0], [2.0, 3.0], [0.0, 1.0], [-1.0, 2.0]],
        {},
        AssertionError,
        id="one_invalid",
    ),
    # all quantiles map to one point beyond the last break: zero interquantile range
    pytest.param([[1.0, 0.001], [1.0, 1e-9]], {}, "finite", id="beyond_last_break"),
    pytest.param(
        POSTERIORS, {"original": [0.5, 1.0, 3.0]}, AssertionError, id="below_first"
    ),
    # NOTE: current behaviour, possibly a bug: `max_shape=0` gives
    # shape -1 (improper) and zero rate
    pytest.param(POSTERIORS, {"max_shape": 0.0}, "finite", id="max_shape_0"),
    pytest.param(POSTERIORS, {"max_shape": 1.0}, "finite", id="max_shape_1"),
    pytest.param(POSTERIORS, {"max_shape": 10.0}, "finite", id="max_shape_10"),
    pytest.param(POSTERIORS, {"max_shape": 1e8}, "finite", id="max_shape_1e8"),
    pytest.param(POSTERIORS, {"max_shape": INF}, "finite", id="max_shape_inf"),
    pytest.param(POSTERIORS, {"max_shape": NAN}, "finite", id="max_shape_nan"),
    *[
        pytest.param(POSTERIORS, {"original": o, "rescaled": r}, ValueError, id=i)
        for i, o, r in [
            ("original_flat", [0.0, 1.0, 1.0], RESCALED),
            ("rescaled_flat", ORIGINAL, [0.0, 2.0, 2.0]),
            ("rescaled_decreasing", ORIGINAL, [0.0, 2.0, 1.0]),
            ("rescaled_nan", ORIGINAL, [0.0, 2.0, NAN]),
        ]
    ],
    # the breaks are checked before the posteriors
    pytest.param(
        [[-5.0, 1.0]],
        {"original": [0.0, 0.0], "rescaled": [0.0, 1.0]},
        ValueError,
        id="bad_breaks_before_posteriors",
    ),
    # a single (zero-length) interval: everything is beyond the last break
    pytest.param(
        POSTERIORS,
        {"original": [0.0], "rescaled": [0.0]},
        ZeroDivisionError,
        id="single_break_zero",
    ),
    pytest.param(
        POSTERIORS, {"original": [0.0], "rescaled": [4.0]}, "finite", id="single_break"
    ),
    pytest.param(POSTERIORS, {"fixed": [0] * 3}, AssertionError, id="fixed_mismatch"),
    pytest.param(
        POSTERIORS,
        {"original": [0.0, 1.0], "rescaled": RESCALED},
        AssertionError,
        id="breaks_mismatch",
    ),
]


def posterior_args(
    posteriors,
    fixed=None,
    original=None,
    rescaled=None,
    quantile_width=0.5,
    max_shape=1000.0,
):
    posteriors = np.asarray(posteriors, dtype=float).reshape(-1, 2)
    fixed = np.full(posteriors.shape[0], False) if fixed is None else b1(*fixed)
    original = f64(*(ORIGINAL if original is None else original))
    rescaled = f64(*(RESCALED if rescaled is None else rescaled))
    return (posteriors, fixed, original, rescaled, quantile_width, max_shape)


class TestPiecewiseScalePosterior:
    @pytest.mark.parametrize(("posteriors", "kwargs", "expected"), POSTERIOR_CASES)
    def test_edge_cases(self, impl, posteriors, kwargs, expected):
        args = posterior_args(posteriors, **kwargs)
        assert_outcome(
            impl.rescaling.piecewise_scale_posterior,
            rescaling.piecewise_scale_posterior,
            args,
            expected,
        )

    @pytest.mark.parametrize("seed", range(3))
    def test_random(self, impl, seed):
        rng = np.random.default_rng(seed)
        k = int(rng.integers(2, 10))
        original = np.concatenate(([0.0], np.cumsum(rng.uniform(0.1, 5.0, k))))
        rescaled = np.concatenate(([0.0], np.cumsum(rng.uniform(0.1, 5.0, k))))
        alpha = rng.uniform(-0.8, 60.0, 15)
        beta = np.exp(rng.uniform(-2, 4, 15)) * (alpha + 1) / (original[-1] * 0.8)
        args = (
            np.column_stack((alpha, beta)),
            rng.random(15) < 0.2,
            original,
            rescaled,
            float(rng.uniform(0.2, 0.9)),
            float(rng.choice([2.0, 100.0, 1000.0])),
        )
        compare(
            impl.rescaling.piecewise_scale_posterior,
            rescaling.piecewise_scale_posterior,
            *args,
        )

    def test_simulated(self, impl, ep):
        nodes_time, likelihoods, nodes_fixed, parents, children = rescale_inputs(ep)
        original, rescaled = rescaling.mutational_timescale(
            nodes_time, likelihoods, nodes_fixed, parents, children, 10
        )
        args = (ep.node_posterior, nodes_fixed, original, rescaled, 0.8, 1000.0)
        compare(
            impl.rescaling.piecewise_scale_posterior,
            rescaling.piecewise_scale_posterior,
            *args,
        )
