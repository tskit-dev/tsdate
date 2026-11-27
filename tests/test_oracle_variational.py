"""
Equivalence of the expectation propagation of ``_tsdate`` with the oracle of
``tsdate.variational``: the EP helpers (``damp``, ``rescale``,
``constraints_are_consistent``), and the methods of ``ExpectationPropagation``,
compared after the same sequence of passes from the same inputs.

Inputs are those of EP on the session fixtures, with constraints and counts edited to
force branches, and tables of boundary and failure cases. Expected values always come
from the oracle. Where the C library reports a division by zero and the oracle does
not raise, the C function alone is checked.
"""

import copy
import re

import _tsdate
import numpy as np
import pytest
import tskit
from oracle import ep_state, variational
from oracle.errors import (
    INVALID_DAMPING,
    INVALID_MAX_SHAPE,
    INVALID_MIN_STEP,
    INVALID_NODE_POSTERIOR,
    NONPOSITIVE_PENALTY,
    ZERO_DIVISION,
)
from oracle.helpers import INF, NAN, compare, f64, kind, log_uniform

from tsdate.variational import (
    LOWER,
    UPPER,
    USE_BLOCK_LIKELIHOOD,
    USE_EDGE_LIKELIHOOD,
    ExpectationPropagation,
)

up = np.nextafter
TS_NAMES = list(ep_state.SPECS)
INCONSISTENT_MESSAGE = (
    "Node age constraints are inconsistent, some descendants have"
    " lower bounds that exceed upper bounds of their ancestors."
)


@pytest.fixture(scope="module")
def inputs(small_ts, small_ts_unphased, small_ts_historical):
    """
    The inputs of EP, by fixture name
    """
    tss = [small_ts, small_ts_unphased, small_ts_historical]
    return {name: ep_state.make_inputs(ts, name) for name, ts in zip(TS_NAMES, tss)}


@pytest.fixture(scope="module")
def started(inputs):
    """
    The oracle's EP two iterations in, by fixture name
    """
    return {name: ep_state.start(variational, ins) for name, ins in inputs.items()}


def compare_runs(ins, step, started=None, iterations=2):
    """
    Run ``step`` on the ``ExpectationPropagation`` of ``_tsdate`` and of the oracle
    after ``iterations`` (``started``, if given, is the oracle's), and compare what it
    returns and the state after it
    """

    def run(ep):
        result = step(ep)
        return result, *ep_state.state(ep)

    def actual():
        return run(ep_state.start(_tsdate, ins, iterations))

    def expected():
        if started is None:
            return run(ep_state.start(variational, ins, iterations))
        return run(copy.deepcopy(started))

    return compare(actual, expected)


def unphased_name(ts_name):
    return ts_name == "small_ts_unphased"


def raises_zero_division():
    """
    ``pytest.raises`` for the library's float division by zero
    """
    message = _tsdate.strerror(ZERO_DIVISION)
    return pytest.raises(_tsdate.LibraryError, match=re.escape(message))


def likelihood_pass(order, unphased, max_shape=1000.0, min_step=0.1):
    """
    A step that runs ``propagate_likelihood``
    """

    def step(ep):
        ep.propagate_likelihood(
            order=order, unphased=unphased, max_shape=max_shape, min_step=min_step
        )

    return step


def prior_pass(free, max_shape=1000.0, em_maxitt=10, em_reltol=1e-8):
    """
    A step that runs ``propagate_prior``
    """

    def step(ep):
        ep.propagate_prior(
            free=free, max_shape=max_shape, em_maxitt=em_maxitt, em_reltol=em_reltol
        )

    return step


def mutations_pass(mutations_edge, unphased):
    """
    A step that runs ``propagate_mutations``
    """

    def step(ep):
        return ep.propagate_mutations(mutations_edge=mutations_edge, unphased=unphased)

    return step


# fmt: off
DAMP_TABLE = [
    # (id, x, y, s, kind); first component test is `1 + x0 - y0 > (1 + x0) * s`,
    # second is `x1 - y1 > x1 * s`. With x = (1, 2) and s = 0.5 the thresholds are
    # y0 = 1 and y1 = 1.
    ("zeros", f64(0, 0), f64(0, 0), 0.5, "finite"),
    ("zeros_invalid_s", f64(0, 0), f64(0, 0), 5.0, "finite"),
    ("zero_y", f64(1, 2), f64(0, 0), 0.5, "finite"),
    ("a_below", f64(1, 2), f64(up(1.0, 0.0), 0), 0.5, "finite"),
    ("a_at", f64(1, 2), f64(1.0, 0), 0.5, "finite"),
    ("a_above", f64(1, 2), f64(up(1.0, 2.0), 0), 0.5, "finite"),
    ("a_damped", f64(1, 2), f64(4, 0), 0.5, "finite"),
    ("a_negative_y", f64(1, 2), f64(-5, 0), 0.5, "finite"),
    ("b_below", f64(1, 2), f64(0, up(1.0, 0.0)), 0.5, "finite"),
    ("b_at", f64(1, 2), f64(0, 1.0), 0.5, "finite"),
    ("b_above", f64(1, 2), f64(0, up(1.0, 2.0)), 0.5, "finite"),
    ("b_damped", f64(1, 2), f64(0, 8), 0.5, "finite"),
    ("b_negative_y", f64(1, 2), f64(0, -5), 0.5, "finite"),
    ("min_is_a", f64(1, 2), f64(4, 2), 0.5, "finite"),
    ("min_is_b", f64(1, 2), f64(2, 8), 0.5, "finite"),
    ("both_negative_y", f64(1, 2), f64(-1, -1), 0.5, "finite"),
    ("tiny_s", f64(1, 2), f64(2, 2), up(0.0, 1.0), "finite"),
    ("s_below_one", f64(1, 2), f64(2, 2), up(1.0, 0.0), "finite"),
    ("x0_just_above_minus_one", f64(up(-1.0, 0.0), 1), f64(0, 0), 0.5, "finite"),
    # NOTE: current behaviour, possibly a bug: `min(a, b)` swallows a
    # NaN in the second component of y, so the damping is silently skipped.
    ("y1_nan", f64(1, 1), f64(0, NAN), 0.5, "finite"),
    ("y1_nan_a_damped", f64(1, 1), f64(8, NAN), 0.5, "finite"),
    ("y0_minus_inf", f64(1, 1), f64(-INF, 0), 0.5, "finite"),
    ("y1_minus_inf", f64(1, 1), f64(0, -INF), 0.5, "finite"),
    # `s` is checked by the callers (direct calls only)
    ("s_zero", f64(1, 1), f64(0, 0), 0.0, INVALID_MIN_STEP),
    ("s_one", f64(1, 1), f64(0, 0), 1.0, INVALID_MIN_STEP),
    ("s_negative", f64(1, 1), f64(0, 0), -0.5, INVALID_MIN_STEP),
    ("s_nan", f64(1, 1), f64(0, 0), NAN, INVALID_MIN_STEP),
    # `0 < x[0] + 1` and `0 < x[1]`: reachable from improper or NaN posteriors
    ("x0_minus_one", f64(-1, 1), f64(0, 0), 0.5, INVALID_NODE_POSTERIOR),
    ("x0_nan", f64(NAN, 1), f64(0, 0), 0.5, INVALID_NODE_POSTERIOR),
    ("x1_zero", f64(1, 0), f64(0, 0), 0.5, INVALID_NODE_POSTERIOR),
    ("x1_negative", f64(1, -1), f64(0, 0), 0.5, INVALID_NODE_POSTERIOR),
    ("x1_nan", f64(1, NAN), f64(0, 0), 0.5, INVALID_NODE_POSTERIOR),
    ("x_zero_y_not", f64(0, 0), f64(1, 1), 0.5, INVALID_NODE_POSTERIOR),
    # `0 < d` fails when d underflows or y[0] is NaN
    ("d_underflow", f64(1, 1e-300), f64(0, 1e300), 0.5, INVALID_DAMPING),
    ("y0_nan", f64(1, 1), f64(NAN, 0), 0.5, INVALID_DAMPING),
]
RESCALE_TABLE = [
    # (id, x, s, kind); branches `1 + x0 > s` and `1 + x0 < 1 / s`
    ("zeros", f64(0, 0), 10.0, "finite"),
    ("zeros_invalid_s", f64(0, 0), 0.0, "finite"),
    ("x0_zero", f64(0, 1), 10.0, "finite"),
    ("inside", f64(5, 1), 10.0, "finite"),
    ("inside_low", f64(-0.5, 1), 2.0, "finite"),
    ("upper_below", f64(up(9.0, 0.0), 1), 10.0, "finite"),
    ("upper_at", f64(9, 1), 10.0, "finite"),
    ("upper_above", f64(up(9.0, 10.0), 1), 10.0, "finite"),
    ("lower_above", f64(up(-0.5, 0.0), 1), 2.0, "finite"),
    ("lower_at", f64(-0.5, 1), 2.0, "finite"),
    ("lower_below", f64(up(-0.5, -1.0), 1), 2.0, "finite"),
    ("lower_far", f64(-0.95, 1), 10.0, "finite"),
    ("infinite_rate", f64(99, INF), 10.0, "finite"),
    ("infinite_shape", f64(INF, 1), 10.0, "finite"),
    # NOTE: current behaviour, possibly a bug: with `max_shape = 1`
    # (allowed by the checks in `propagate_*`) the factor is 0 and wipes out the
    # posterior.
    ("s_one_upper", f64(1, 1), 1.0, "finite"),
    ("s_one_lower", f64(-0.5, 1), 1.0, "finite"),
    # reachable as for `damp`
    ("x0_minus_one", f64(-1, 1), 10.0, INVALID_NODE_POSTERIOR),
    ("x0_nan", f64(NAN, 1), 10.0, INVALID_NODE_POSTERIOR),
    ("x1_zero", f64(1, 0), 10.0, INVALID_NODE_POSTERIOR),
    ("x1_negative", f64(1, -1), 10.0, INVALID_NODE_POSTERIOR),
    ("x1_nan", f64(1, NAN), 10.0, INVALID_NODE_POSTERIOR),
]
# fmt: on


class TestDamp:
    @pytest.mark.parametrize(
        ("x", "y", "s", "expected_kind"),
        [pytest.param(*row[1:], id=row[0]) for row in DAMP_TABLE],
    )
    def test_table(self, x, y, s, expected_kind):
        result = compare(_tsdate.damp, variational.damp, *x, *y, s)
        assert kind(result) == expected_kind

    @pytest.mark.parametrize("seed", range(8))
    def test_random(self, seed):
        rng = np.random.default_rng(seed)
        x = f64(rng.uniform(-0.9, 30), log_uniform(rng, -4, -1))
        y = f64(rng.uniform(-5, 30), log_uniform(rng, -4, -1))
        s = float(rng.uniform(0.05, 0.95))
        compare(_tsdate.damp, variational.damp, *x, *y, s)

    @pytest.mark.parametrize("which", [0, 1])
    def test_infinite_posterior_zero_message(self, which):
        # C reports the division of inf by 0.0; the oracle, on the NumPy scalars of
        # the passes, returns d = 1. Needs an infinite posterior, which is
        # unreachable without a prior failure.
        x = f64(1, 1)
        x[which] = INF
        with raises_zero_division():
            _tsdate.damp(*x, 0.0, 0.0, 0.5)


class TestRescale:
    @pytest.mark.parametrize(
        ("x", "s", "expected_kind"),
        [pytest.param(*row[1:], id=row[0]) for row in RESCALE_TABLE],
    )
    def test_table(self, x, s, expected_kind):
        result = compare(_tsdate.rescale, variational.rescale, *x, s)
        assert kind(result) == expected_kind

    @pytest.mark.parametrize("seed", range(8))
    def test_random(self, seed):
        rng = np.random.default_rng(seed)
        x = f64(rng.uniform(-0.9, 3000), log_uniform(rng, -4, -1))
        s = float(rng.choice([10.0, 100.0, 1000.0]))
        compare(_tsdate.rescale, variational.rescale, *x, s)

    def test_zero_shape_offset_cap_below_one(self):
        # C reports the division by zero; the oracle, on NumPy scalars, returns
        # -inf. Direct call only: `propagate_*` check `max_shape >= 1`.
        with raises_zero_division():
            _tsdate.rescale(0.0, 1.0, 0.5)


class TestConstraintsAreConsistent:
    @staticmethod
    def check(constraints, parents, children, valid):
        args = (
            np.array(constraints, dtype=np.float64),
            np.array(parents, dtype=np.int32),
            np.array(children, dtype=np.int32),
        )
        func = _tsdate.constraints_are_consistent
        result = compare(func, variational.constraints_are_consistent, *args)
        assert result.value is valid

    @pytest.mark.parametrize(
        ("constraints", "parents", "children", "valid"),
        [
            pytest.param(*row[1:], id=row[0])
            for row in [
                ("no_edges", [[0, 1], [0, 1]], [], [], True),
                ("free_nodes", [[0, 0], [0, INF]], [1], [0], True),
                ("lower_below_upper", [[0, 5], [3, 9]], [1], [0], True),
                # boundary: the parent's upper bound equals the child's lower bound
                ("equal_bounds", [[5, 5], [0, 5]], [1], [0], True),
                ("equal_samples", [[5, 5], [5, 5]], [1], [0], True),
                ("child_just_above", [[up(5.0, 6.0), 6], [0, 5]], [1], [0], False),
                ("parent_just_below", [[5, 5], [0, up(5.0, 4.0)]], [1], [0], False),
                # the max lower bound propagates up a chain, and from every child
                ("chain_valid", [[5, 5], [0, 9], [0, 9]], [1, 2], [0, 1], True),
                ("chain_invalid", [[5, 5], [0, INF], [0, 4]], [1, 2], [0, 1], False),
                ("two_children", [[0, 0], [7, 7], [0, 6]], [2, 2], [0, 1], False),
                ("sample_ancestor", [[0, 0], [10, 10]], [1], [0], True),
                ("sample_descendant_older", [[10, 10], [0, 0]], [1], [0], False),
                # NOTE: current behaviour, possibly a bug: the check is
                # a single pass in edge order, so edges must be sorted children-first.
                (
                    "chain_edges_reversed",
                    [[5, 5], [0, INF], [0, 4]],
                    [2, 1],
                    [1, 0],
                    True,
                ),
                # NaN comparisons are false: NaN bounds are never inconsistent
                ("nan_lower", [[NAN, NAN], [0, 5]], [1], [0], True),
                ("nan_upper", [[5, 5], [0, NAN]], [1], [0], True),
            ]
        ],
    )
    def test_table(self, constraints, parents, children, valid):
        self.check(constraints, parents, children, valid)

    @pytest.mark.parametrize("ts_name", ["small_ts", "small_ts_historical"])
    @pytest.mark.parametrize("gap", ["equal", "one_ulp"])
    def test_real_edges(self, inputs, ts_name, gap):
        # the lower bound of a child equals the upper bound of its parent, or exceeds it
        ins = inputs[ts_name]
        constraints = ins.node_constraints.copy()
        p, c = ins.edge_parents[-1], ins.edge_children[-1]
        constraints[c, LOWER] = 500.0
        constraints[p, UPPER] = 500.0 if gap == "equal" else up(500.0, 0.0)
        valid = gap == "equal"
        self.check(constraints, ins.edge_parents, ins.edge_children, valid)

    def test_production_raises(self):
        constraints = np.array([[10.0, 10.0], [0.0, 0.0]])
        edge = np.array([1], dtype=np.int32), np.array([0], dtype=np.int32)
        check = ExpectationPropagation._check_valid_constraints
        with pytest.raises(ValueError, match=re.escape(INCONSISTENT_MESSAGE)):
            check(constraints, *edge)


class TestFactors:
    @pytest.mark.parametrize("ts_name", TS_NAMES)
    def test_rescale_factors(self, inputs, started, ts_name):
        # a pass with a small cap leaves scales that are not one
        ins = inputs[ts_name]
        unphased = unphased_name(ts_name)
        order = ins.block_order if unphased else ins.edge_order

        def step(ep):
            likelihood_pass(order, unphased, max_shape=3.0)(ep)
            ep.rescale_factors()
            return ep.assemble_factors()

        compare_runs(ins, step, started[ts_name])

    @pytest.mark.parametrize("ts_name", TS_NAMES)
    def test_assemble_factors(self, inputs, started, ts_name):
        compare_runs(inputs[ts_name], lambda ep: ep.assemble_factors(), started[ts_name])


class TestPropagateLikelihood:
    SUBSET = slice(0, 400)  # an initial portion of the pass keeps the oracle quick

    @staticmethod
    def order(ins, unphased, subset=None):
        order = ins.block_order if unphased else ins.edge_order
        return order if subset is None else order[subset]

    @pytest.mark.parametrize(
        ("ts_name", "max_shape", "min_step"),
        [
            ("small_ts", 1000.0, 0.1),
            ("small_ts_unphased", 1000.0, 0.1),
            ("small_ts_historical", 1000.0, 0.1),
            ("small_ts", 10.0, 0.5),
            ("small_ts_unphased", 10.0, 0.5),
        ],
    )
    def test_natural(self, inputs, started, ts_name, max_shape, min_step):
        ins = inputs[ts_name]
        unphased = unphased_name(ts_name)
        counts = ins.branch_counts(unphased)
        assert counts["both_free"] > 0
        assert counts["both_fixed"] == counts["fixed_parent"] == 0
        step = likelihood_pass(self.order(ins, unphased), unphased, max_shape, min_step)
        compare_runs(ins, step, started[ts_name])

    @pytest.mark.parametrize("ts_name", TS_NAMES)
    def test_forced_fixed_nodes(self, inputs, ts_name):
        # every branch: both nodes fixed (skipped), one fixed, both free
        unphased = unphased_name(ts_name)
        ins = ep_state.fixed_internal(inputs[ts_name], unphased)
        counts = ins.branch_counts(unphased)
        assert all(v > 0 for k, v in counts.items() if k != "twin" or unphased)
        step = likelihood_pass(self.order(ins, unphased), unphased)
        compare_runs(ins, step, iterations=1)

    @pytest.mark.parametrize("kind_of_order", ["empty", "partial_and_repeated"])
    def test_order(self, inputs, started, kind_of_order):
        # the order array is arbitrary: empty, or any subset with repeats
        ins = inputs["small_ts"]
        if kind_of_order == "empty":
            order = np.zeros(0, dtype=np.int32)
        else:
            order = ins.edge_order[100:140]
            order = np.concatenate((order, order[::-1], order[:5])).astype(np.int32)
        compare_runs(ins, likelihood_pass(order, False), started["small_ts"])

    @pytest.mark.parametrize(
        "max_shape",
        # NOTE: current behaviour, possibly a bug: a cap of exactly 1
        # zeroes posteriors and scales (and then every later update is NaN)
        [1.0, up(1.0, 2.0), 3.0, 1e300],
        ids=lambda v: f"max_shape={v:g}",
    )
    @pytest.mark.parametrize("ts_name", ["small_ts", "small_ts_unphased"])
    def test_max_shape(self, inputs, started, ts_name, max_shape):
        ins = inputs[ts_name]
        unphased = unphased_name(ts_name)
        order = self.order(ins, unphased, self.SUBSET)
        step = likelihood_pass(order, unphased, max_shape=max_shape)
        compare_runs(ins, step, started[ts_name])

    @pytest.mark.parametrize(
        "min_step",
        [up(0.0, 1.0), 0.1, 0.99, 1 - 1e-9],
        ids=lambda v: f"min_step={v:g}",
    )
    @pytest.mark.parametrize("ts_name", ["small_ts", "small_ts_unphased"])
    def test_min_step(self, inputs, started, ts_name, min_step):
        ins = inputs[ts_name]
        unphased = unphased_name(ts_name)
        order = self.order(ins, unphased, self.SUBSET)
        # a small cap makes passes difficult, so that damping is exercised
        step = likelihood_pass(order, unphased, max_shape=3.0, min_step=min_step)
        compare_runs(ins, step, started[ts_name])

    @pytest.mark.parametrize("multiplier", [1e12, 1e300], ids=lambda v: f"y*{v:g}")
    def test_extreme_counts(self, inputs, multiplier):
        # 1e12: projection failures are skipped (NaN lognorm). 1e300: a posterior
        # becomes NaN and the next `damp` fails its check `0 < x[0] + 1`,
        # reachable from real data (~1e300 mutations on an edge).
        ins = inputs["small_ts"].copy()
        ins.edge_likelihoods = ins.edge_likelihoods * [multiplier, 1.0]
        order = ins.edge_order[: self.SUBSET.stop]
        compare_runs(ins, likelihood_pass(order, False), iterations=0)

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            pytest.param(
                dict(max_shape=up(1.0, 0.0)), INVALID_MAX_SHAPE, id="max_shape_below_one"
            ),
            pytest.param(dict(max_shape=0.0), INVALID_MAX_SHAPE, id="max_shape_zero"),
            pytest.param(dict(max_shape=NAN), INVALID_MAX_SHAPE, id="max_shape_nan"),
            pytest.param(dict(min_step=0.0), INVALID_MIN_STEP, id="min_step_zero"),
            pytest.param(dict(min_step=1.0), INVALID_MIN_STEP, id="min_step_one"),
            pytest.param(dict(min_step=-0.1), INVALID_MIN_STEP, id="min_step_negative"),
            pytest.param(dict(min_step=1.5), INVALID_MIN_STEP, id="min_step_above_one"),
            pytest.param(dict(min_step=NAN), INVALID_MIN_STEP, id="min_step_nan"),
        ],
    )
    def test_invalid_parameters(self, inputs, kwargs, expected):
        # reachable from user input (`max_shape`, `min_step` of `iterate`/`infer`):
        # parameter errors in the C library, not internal invariants
        ins = inputs["small_ts"]
        step = likelihood_pass(ins.edge_order[:10], False, **kwargs)
        result = compare_runs(ins, step, iterations=0)
        assert kind(result) == expected


class TestPropagatePrior:
    @pytest.mark.parametrize("ts_name", TS_NAMES)
    def test_natural(self, inputs, started, ts_name):
        ins = inputs[ts_name]
        assert ins.unconstrained_roots.any()
        compare_runs(ins, prior_pass(ins.unconstrained_roots), started[ts_name])

    @pytest.mark.parametrize(
        ("em_maxitt", "em_reltol"),
        # `em_maxitt`: no iteration (-1), one (0), the cap binding (3), convergence
        # binding (1000); `em_reltol`: run to the cap (0), tight, loose, NaN (no
        # iteration)
        [(m, r) for m in [-1, 0, 3, 1000] for r in [0.0, 1e-8, 0.5, NAN]],
        ids=lambda v: f"{v:g}",
    )
    def test_em(self, inputs, started, em_maxitt, em_reltol):
        ins = inputs["small_ts"]
        step = prior_pass(
            ins.unconstrained_roots,
            em_maxitt=em_maxitt,
            em_reltol=em_reltol,
            max_shape=1e300,
        )
        compare_runs(ins, step, started["small_ts"])

    @pytest.mark.parametrize("ts_name", ["small_ts", "small_ts_historical"])
    @pytest.mark.parametrize(
        "max_shape",
        # NOTE: current behaviour, possibly a bug: a cap of exactly 1
        # zeroes the posteriors and scales of the free nodes
        [1.0, 1.5, 1e300],
        ids=lambda v: f"max_shape={v:g}",
    )
    def test_max_shape(self, inputs, started, ts_name, max_shape):
        ins = inputs[ts_name]
        step = prior_pass(ins.unconstrained_roots, max_shape=max_shape)
        compare_runs(ins, step, started[ts_name])

    @pytest.mark.parametrize(
        ("ts_name", "free", "kwargs"),
        [
            ("small_ts", "single_root", {}),
            # any node set with proper cavities works, not only roots
            ("small_ts_unphased", "internal", {}),
            ("small_ts", "none", {}),
            # no free nodes returns before looking at anything else
            ("small_ts", "none", dict(em_maxitt=-5, em_reltol=NAN, max_shape=1.0)),
        ],
        ids=["single_root", "internal", "none", "none_odd_arguments"],
    )
    def test_free_nodes(self, inputs, started, ts_name, free, kwargs):
        ins = inputs[ts_name]
        mask = np.zeros(ins.node_constraints.shape[0], dtype=bool)
        if free == "single_root":
            mask[np.flatnonzero(ins.unconstrained_roots)[0]] = True
        elif free == "internal":
            mask = ~ins.fixed & ~ins.roots | ins.unconstrained_roots
        compare_runs(ins, prior_pass(mask, **kwargs), started[ts_name])

    @pytest.mark.parametrize(
        ("case", "expected"),
        [
            # `penalty > 0` fails when the mean of shape / rate is infinite or
            # negative: free nodes whose cavity has zero rate (fixed leaves)
            ("all_nodes", NONPOSITIVE_PENALTY),
            ("leaves", NONPOSITIVE_PENALTY),
            # reachable from user input
            ("max_shape_below_one", INVALID_MAX_SHAPE),
            ("max_shape_zero", INVALID_MAX_SHAPE),
            ("max_shape_nan", INVALID_MAX_SHAPE),
        ],
    )
    def test_errors(self, inputs, started, case, expected):
        ins = inputs["small_ts"]
        free = ins.unconstrained_roots
        kwargs = {}
        if case == "all_nodes":
            free = np.ones_like(free)
        elif case == "leaves":
            free = ins.leaves
        else:
            kwargs["max_shape"] = {
                "below_one": up(1.0, 0.0),
                "zero": 0.0,
                "nan": NAN,
            }[case[len("max_shape_") :]]
        result = compare_runs(ins, prior_pass(free, **kwargs), started["small_ts"])
        assert kind(result) == expected

    def test_penalty_error_reachable_from_ep(self, inputs):
        # NOTE: current behaviour, possibly a bug: a likelihood pass in
        # which every update is skipped (a huge mutational target) leaves the roots
        # with a zero posterior, so that `iterate(regularise=True)` fails the
        # `penalty > 0` check (an error, not NaN).
        ins = inputs["small_ts"].copy()
        ins.edge_likelihoods = ins.edge_likelihoods * [1.0, 1e300]

        def step(ep):
            likelihood_pass(ins.edge_order, False)(ep)
            prior_pass(ins.unconstrained_roots)(ep)

        result = compare_runs(ins, step, iterations=0)
        assert kind(result) == NONPOSITIVE_PENALTY


class TestPropagateMutations:
    @pytest.mark.parametrize(
        ("ts_name", "unphased"),
        [
            ("small_ts", False),
            ("small_ts_historical", False),
            ("small_ts_unphased", False),
            ("small_ts_unphased", True),
        ],
    )
    def test_natural(self, inputs, started, ts_name, unphased):
        ins = inputs[ts_name]
        step = mutations_pass(ins.mutations_edge(unphased), unphased)
        compare_runs(ins, step, started[ts_name])

    @pytest.mark.parametrize(
        ("ts_name", "unphased"),
        [
            ("small_ts", False),
            ("small_ts_historical", False),
            ("small_ts_unphased", True),
        ],
    )
    def test_forced_fixed_nodes(self, inputs, ts_name, unphased):
        # mutations on edges of every kind: both nodes fixed, one of the two fixed
        ins = ep_state.fixed_internal(inputs[ts_name], unphased)
        counts = ins.branch_counts(unphased)
        assert all(counts[k] > 0 for k in ("both_fixed", "fixed_parent", "fixed_child"))
        step = mutations_pass(ins.mutations_edge(unphased), unphased)
        compare_runs(ins, step, iterations=1)

    @pytest.mark.parametrize("case", ["null_edges", "subset_and_repeats"])
    def test_mutation_selection(self, inputs, started, case):
        # mutations above the root have edge NULL and get NaN posteriors; the
        # mutations are any edges, with repeats
        ins = inputs["small_ts"]
        edges = ins.mutations_edge(False).copy()
        if case == "null_edges":
            edges[::7] = tskit.NULL
        else:
            edges = edges[[5, 3, 3, 100, 5]]
        compare_runs(ins, mutations_pass(edges, False), started["small_ts"])


class TestMultiSweep:
    """
    Whole sweeps from the initial state: the node posteriors, likelihood
    normalisers and mutation posteriors of every sweep are identical to the oracle's.
    """

    CASES = {
        "small_ts": ("small_ts", {}),
        "small_ts_unphased": ("small_ts_unphased", {}),
        "small_ts_historical_cap": (
            "small_ts_historical",
            dict(max_shape=10.0, min_step=0.5, em_maxitt=3),
        ),
    }

    @staticmethod
    def run(module, ins, sweeps, kwargs):
        ep = module.ExpectationPropagation(**ins.kwargs())
        states = []
        for _ in range(sweeps):
            ep_state.iterate(ep, ins, **kwargs)
            states.append(ep_state.state(ep))
        mutations = [
            ep.propagate_mutations(
                mutations_edge=ins.mutations_edge(unphased), unphased=unphased
            )
            for unphased in (USE_BLOCK_LIKELIHOOD, USE_EDGE_LIKELIHOOD)
        ]
        return states, mutations

    @pytest.mark.parametrize("case", CASES)
    def test_sweeps(self, inputs, case):
        name, kwargs = self.CASES[case]
        ins = inputs[name]
        actual = self.run(_tsdate, ins, 4, kwargs)
        expected = self.run(variational, ins, 4, kwargs)
        for a, e in zip(actual[0], expected[0]):
            for x, y in zip(a, e):  # node posterior, edge and block lognorm
                np.testing.assert_array_equal(x, y)
        for a, e in zip(actual[1], expected[1]):  # mutation posterior, phase
            for x, y in zip(a, e):
                np.testing.assert_array_equal(x, y)
