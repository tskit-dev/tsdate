"""
Equivalence of the expectation propagation functions of ``_tsdate`` with the oracle of
``tsdate.variational``: the EP helpers (``damp``, ``rescale``,
``rescale_factors``, ``constraints_are_consistent``) and the passes
``propagate_likelihood``, ``propagate_prior`` and ``propagate_mutations``, plus whole
sweeps. The EP state is a tuple of factor arrays, which the passes update in place.

Inputs are snapshots of real EP state (``oracle.ep_state``) on the session fixtures,
with constraints edited to force branches, one-edge snapshots for each branch, and
tables of boundary and failure cases. Expected values always come from the oracle.
Where the C library reports a division by zero and the oracle does not raise, the C
function alone is checked.
"""

import re
from functools import partial

import _tsdate
import numpy as np
import pytest
import tskit
from oracle import ep_state, variational
from oracle.ep_state import BRANCHES, SNAPSHOT_SPECS, tiny_snapshot
from oracle.errors import (
    INVALID_DAMPING,
    INVALID_MAX_SHAPE,
    INVALID_MIN_STEP,
    INVALID_NODE_POSTERIOR,
    NONPOSITIVE_PENALTY,
    TWIN_EDGE_IN_PHASED_PASS,
    ZERO_DIVISION,
)
from oracle.helpers import (
    INF,
    NAN,
    assert_same_outcome,
    assert_same_values,
    compare,
    f64,
    kind,
    log_uniform,
    outcome,
)

from tsdate.variational import (
    LOWER,
    MIXPRIOR,
    TINY,
    UPPER,
    ExpectationPropagation,
)

up = np.nextafter
TS_NAMES = list(SNAPSHOT_SPECS)
INCONSISTENT_MESSAGE = (
    "Node age constraints are inconsistent, some descendants have"
    " lower bounds that exceed upper bounds of their ancestors."
)


@pytest.fixture(scope="module")
def snaps(small_ts, small_ts_unphased, small_ts_historical):
    """
    Snapshots of EP state two iterations in, by fixture name
    """
    tss = [small_ts, small_ts_unphased, small_ts_historical]
    return {name: ep_state.make_snapshot(ts, name) for name, ts in zip(TS_NAMES, tss)}


@pytest.fixture(scope="module")
def fresh(small_ts, small_ts_unphased, small_ts_historical):
    """
    Snapshots of the initial EP state, by fixture name
    """
    tss = [small_ts, small_ts_unphased, small_ts_historical]
    return {
        name: ep_state.make_snapshot(ts, name, iterations=0)
        for name, ts in zip(TS_NAMES, tss)
    }


@pytest.fixture(scope="module")
def scaled_snap(snaps):
    """
    ``small_ts`` after a likelihood pass with a small cap, so that scales are not 1
    """
    snap = snaps["small_ts"]
    state = ep_state.run_likelihood(variational, snap, False, max_shape=3.0)
    return snap.after_pass(state)


def compare_runs(runner, snap, *args, **kwargs):
    """
    Run ``runner`` (an ``ep_state.run_*``) with ``_tsdate`` and with the oracle
    """
    actual = partial(runner, _tsdate)
    expected = partial(runner, variational)
    return compare(actual, expected, snap, *args, **kwargs)


def assert_run_kind(runner, snap, expected_kind, *args, **kwargs):
    result = compare_runs(runner, snap, *args, **kwargs)
    assert kind(result) == expected_kind


def unphased_name(ts_name):
    return ts_name == "small_ts_unphased"


def raises_zero_division():
    """
    ``pytest.raises`` for the library's float division by zero
    """
    message = _tsdate.strerror(ZERO_DIVISION)
    return pytest.raises(_tsdate.LibraryError, match=re.escape(message))


class TestRescaleFactors:
    @pytest.mark.parametrize("ts_name", ["small_ts", "small_ts_unphased"])
    @pytest.mark.parametrize("scale", ["seed0", "seed1", 1e-160, 1e100])
    def test_rescale_factors(self, snaps, ts_name, scale):
        snap = snaps[ts_name].copy()
        num_nodes = snap.node_posterior.shape[0]
        if isinstance(scale, str):
            rng = np.random.default_rng(int(scale[-1]))
            scale = rng.uniform(0.5, 2.0, num_nodes)
        snap.fields["scale"][:] = scale
        actual = snap.factors()
        expected = snap.factors()
        _tsdate.rescale_factors(actual)
        variational.rescale_factors(expected)
        assert_same_values(actual[:4], expected[:4])

    def test_rescale_factors_twice(self, snaps):
        # the scale is reset to 1, so a second call changes nothing
        snap = snaps["small_ts"]
        factors = snap.factors()
        _tsdate.rescale_factors(factors)
        once = [f.copy() for f in factors[:3]]
        _tsdate.rescale_factors(factors)
        for before, after in zip(once, factors[:3]):
            assert np.array_equal(before, after)

    @pytest.mark.parametrize("ts_name", TS_NAMES)
    def test_assemble_factors(self, snaps, scaled_snap, ts_name):
        snap = scaled_snap if ts_name == "small_ts" else snaps[ts_name]
        actual = _tsdate.assemble_factors(snap.factors())
        expected = variational.assemble_factors(snap.factors())
        assert_same_outcome(outcome(lambda: actual), outcome(lambda: expected))


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
    def test_real_edges(self, snaps, ts_name, gap):
        # the lower bound of a child equals the upper bound of its parent, or exceeds it
        snap = snaps[ts_name]
        constraints = snap.node_constraints.copy()
        p, c = snap.edge_parents[-1], snap.edge_children[-1]
        constraints[c, LOWER] = 500.0
        constraints[p, UPPER] = 500.0 if gap == "equal" else up(500.0, 0.0)
        valid = gap == "equal"
        self.check(constraints, snap.edge_parents, snap.edge_children, valid)

    def test_production_raises(self):
        constraints = np.array([[10.0, 10.0], [0.0, 0.0]])
        edge = np.array([1], dtype=np.int32), np.array([0], dtype=np.int32)
        check = ExpectationPropagation._check_valid_constraints
        with pytest.raises(ValueError, match=re.escape(INCONSISTENT_MESSAGE)):
            check(constraints, *edge)


class TestPropagateLikelihoodBranches:
    """
    One edge (block) in each branch, with valid and failing projections
    """

    CASES = {
        "valid": dict(),
        "damped": dict(cavity=ep_state.DAMPED_CAVITY, message=ep_state.DAMPED_MESSAGE),
        # NOTE: current behaviour, possibly a bug: a failed projection
        # is not reported; the posterior becomes the cavity and `lognorm` is NaN.
        "projection_fails": dict(likelihood=(1.0, 1e300)),
    }

    @pytest.mark.parametrize("case", CASES)
    @pytest.mark.parametrize("branch", [b for b in BRANCHES if "both_fixed" not in b])
    def test_single_edge(self, branch, case):
        snap = tiny_snapshot(branch, **self.CASES[case])
        kwargs = dict(max_shape=1e300)  # no posterior rescaling
        if "sideways" in branch and case == "projection_fails":
            # `sideways_moments` divides by zero for an astronomically large mutation
            # target: C reports it where the oracle returns NaN and skips.
            with raises_zero_division():
                ep_state.run_likelihood(_tsdate, snap, snap.unphased, **kwargs)
        else:
            compare_runs(ep_state.run_likelihood, snap, snap.unphased, **kwargs)

    def test_twin_edge_in_phased_pass(self):
        # p == c is only valid for blocks; `twin_projection` checks `unphased`.
        # Direct call only: edges never have p == c in a tree sequence.
        snap = tiny_snapshot("twin")
        result = compare_runs(ep_state.run_likelihood, snap, False)
        assert kind(result) == TWIN_EDGE_IN_PHASED_PASS


class TestPropagateLikelihood:
    SUBSET = slice(0, 400)  # an initial portion of the pass keeps the oracle quick

    @staticmethod
    def order(snap, unphased, subset=None):
        order = snap.block_order if unphased else snap.edge_order
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
    def test_natural(self, snaps, ts_name, max_shape, min_step):
        snap = snaps[ts_name]
        unphased = unphased_name(ts_name)
        counts = snap.branch_counts(unphased)
        assert counts["both_free"] > 0
        assert counts["both_fixed"] == counts["fixed_parent"] == 0
        compare_runs(
            ep_state.run_likelihood,
            snap,
            unphased,
            max_shape=max_shape,
            min_step=min_step,
        )

    @pytest.mark.parametrize(
        "ts_name", ["small_ts", "small_ts_unphased", "small_ts_historical"]
    )
    def test_forced_fixed_nodes(self, snaps, ts_name):
        # every branch: both nodes fixed (skipped), one fixed, both free
        unphased = unphased_name(ts_name)
        snap = ep_state.fixed_internal_snapshot(snaps[ts_name], unphased)
        counts = snap.branch_counts(unphased)
        assert all(v > 0 for k, v in counts.items() if k != "twin" or unphased)
        compare_runs(ep_state.run_likelihood, snap, unphased)

    @pytest.mark.parametrize("kind_of_order", ["empty", "partial_and_repeated"])
    def test_order(self, snaps, kind_of_order):
        # the order array is arbitrary: empty, or any subset with repeats
        snap = snaps["small_ts"]
        if kind_of_order == "empty":
            order = np.zeros(0, dtype=np.int32)
        else:
            order = snap.edge_order[100:140]
            order = np.concatenate((order, order[::-1], order[:5])).astype(np.int32)
        compare_runs(ep_state.run_likelihood, snap, False, order=order)

    @pytest.mark.parametrize(
        "max_shape",
        # NOTE: current behaviour, possibly a bug: a cap of exactly 1
        # zeroes posteriors and scales (and then every later update is NaN)
        [1.0, up(1.0, 2.0), 3.0, 1e300],
        ids=lambda v: f"max_shape={v:g}",
    )
    @pytest.mark.parametrize("ts_name", ["small_ts", "small_ts_unphased"])
    def test_max_shape(self, snaps, ts_name, max_shape):
        snap = snaps[ts_name]
        unphased = unphased_name(ts_name)
        order = self.order(snap, unphased, self.SUBSET)
        compare_runs(
            ep_state.run_likelihood,
            snap,
            unphased,
            max_shape=max_shape,
            order=order,
        )

    @pytest.mark.parametrize(
        "min_step",
        [up(0.0, 1.0), 0.1, 0.99, 1 - 1e-9],
        ids=lambda v: f"min_step={v:g}",
    )
    @pytest.mark.parametrize("ts_name", ["small_ts", "small_ts_unphased"])
    def test_min_step(self, snaps, ts_name, min_step):
        snap = snaps[ts_name]
        unphased = unphased_name(ts_name)
        order = self.order(snap, unphased, self.SUBSET)
        # a small cap makes passes difficult, so that damping is exercised
        compare_runs(
            ep_state.run_likelihood,
            snap,
            unphased,
            min_step=min_step,
            max_shape=3.0,
            order=order,
        )

    @pytest.mark.parametrize("multiplier", [1e12, 1e300], ids=lambda v: f"y*{v:g}")
    def test_extreme_counts(self, snaps, multiplier):
        # 1e12: projection failures are skipped (NaN lognorm). 1e300: a posterior
        # becomes NaN and the next `damp` fails its check `0 < x[0] + 1`,
        # reachable from real data (~1e300 mutations on an edge).
        snap = snaps["small_ts"]
        likelihoods = snap.edge_likelihoods * [multiplier, 1.0]
        order = snap.edge_order[: self.SUBSET.stop]
        compare_runs(
            ep_state.run_likelihood,
            snap,
            False,
            order=order,
            likelihoods=likelihoods,
        )

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
    def test_invalid_parameters(self, snaps, kwargs, expected):
        # reachable from user input (`max_shape`, `min_step` of `iterate`/`infer`):
        # parameter errors in the C library, not internal invariants
        snap = snaps["small_ts"]
        order = snap.edge_order[:10]
        assert_run_kind(
            ep_state.run_likelihood,
            snap,
            expected,
            False,
            order=order,
            **kwargs,
        )

    @pytest.mark.parametrize(
        "change",
        ["children", "likelihoods", "constraints", "factors_larger"],
    )
    def test_shape_mismatches(self, snaps, change):
        # direct calls only: these are checked while parsing arguments in C
        snap = snaps["small_ts"]
        args, lognorm = ep_state.likelihood_args(snap, False)
        order, parents, children, likelihoods = args
        constraints = snap.node_constraints
        if change == "children":
            children = children[:-1]
        elif change == "likelihoods":
            likelihoods = likelihoods[:-1]
        elif change == "constraints":
            constraints = constraints[:-1]
        else:  # factors have more edges than the arrays
            parents, children = parents[:-1], children[:-1]
            likelihoods = likelihoods[:-1]

        def call(module):
            module.propagate_likelihood(
                order,
                parents,
                children,
                likelihoods,
                constraints,
                snap.node_posterior.copy(),
                snap.factors(),
                lognorm.copy(),
                1000.0,
                0.1,
                ep_state.USE_EDGE_LIKELIHOOD,
            )

        result = compare(
            lambda: call(_tsdate),
            lambda: call(variational),
        )
        assert kind(result) is ValueError


class TestPropagateLikelihoodTinyScale:
    """
    `propagate_likelihood` calls `rescale_factors` (mid-pass) whenever the scale of
    either node of the next edge is below `TINY` (strict), even if the edge is skipped
    because both nodes are fixed.
    """

    @staticmethod
    def shrunk(snap, role, scale):
        """
        Equivalent state with the scale of one node tiny, first used mid-pass
        """
        snap = snap.copy()
        nodes = snap.edge_parents if role == "parent" else snap.edge_children
        first = {}
        for pos, i in enumerate(snap.edge_order):
            first.setdefault(nodes[i], pos)
        target = len(snap.edge_order) // 3
        node = min(first, key=lambda n: abs(first[n] - target))
        snap.unscale([node], scale)
        return snap

    @pytest.mark.parametrize(
        ("role", "scale"),
        [
            ("parent", TINY / 1e6),
            ("child", TINY / 1e6),
            ("child", up(TINY, 0.0)),
            ("child", TINY),
            ("parent", 2 * TINY),
        ],
        ids=[
            "parent-below",
            "child-below",
            "child-just-below",
            "child-at",
            "parent-above",
        ],
    )
    def test_threshold(self, snaps, role, scale):
        snap = self.shrunk(snaps["small_ts"], role, scale)
        order = snap.edge_order[: len(snap.edge_order) // 2]
        compare_runs(ep_state.run_likelihood, snap, False, order=order)

    def test_skipped_fixed_edge(self, snaps):
        # a fixed sample with a tiny scale whose edge is skipped (both nodes fixed)
        base = snaps["small_ts_historical"].copy()
        sample = np.flatnonzero(base.fixed)[0]
        edge = np.flatnonzero(base.edge_children == sample)[0]
        base.fix_nodes([base.edge_parents[edge]])
        assert base.branch_counts(False)["both_fixed"] > 0
        base.unscale([sample], TINY / 1e6)
        compare_runs(ep_state.run_likelihood, base, False)


class TestPropagatePrior:
    @pytest.mark.parametrize("ts_name", TS_NAMES)
    def test_natural(self, snaps, ts_name):
        snap = snaps[ts_name]
        assert snap.unconstrained_roots.any()
        compare_runs(ep_state.run_prior, snap, snap.unconstrained_roots)

    @pytest.mark.parametrize(
        ("scaled", "em_maxitt", "em_reltol"),
        # `em_maxitt`: no iteration (-1), one (0), the cap binding (3), convergence
        # binding (1000); `em_reltol`: run to the cap (0), tight, loose, NaN (no
        # iteration)
        [(False, m, r) for m in [-1, 0, 3, 1000] for r in [0.0, 1e-8, 0.5, NAN]]
        + [(True, 0, 1e-8), (True, 3, 1e-8), (True, 1000, 0.5)],
        ids=lambda v: f"{v}" if isinstance(v, bool) else f"{v:g}",
    )
    def test_em(self, snaps, scaled_snap, scaled, em_maxitt, em_reltol):
        snap = scaled_snap if scaled else snaps["small_ts"]
        compare_runs(
            ep_state.run_prior,
            snap,
            snap.unconstrained_roots,
            em_maxitt=em_maxitt,
            em_reltol=em_reltol,
            max_shape=1e300,
        )

    @pytest.mark.parametrize("ts_name", ["small_ts", "small_ts_historical"])
    @pytest.mark.parametrize(
        "max_shape",
        # NOTE: current behaviour, possibly a bug: a cap of exactly 1
        # zeroes the posteriors and scales of the free nodes
        [1.0, 1.5, 1e300],
        ids=lambda v: f"max_shape={v:g}",
    )
    def test_max_shape(self, snaps, ts_name, max_shape):
        snap = snaps[ts_name]
        compare_runs(
            ep_state.run_prior,
            snap,
            snap.unconstrained_roots,
            max_shape=max_shape,
        )

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
    def test_free_nodes(self, snaps, ts_name, free, kwargs):
        snap = snaps[ts_name]
        mask = np.zeros(snap.node_posterior.shape[0], dtype=bool)
        if free == "single_root":
            mask[np.flatnonzero(snap.unconstrained_roots)[0]] = True
        elif free == "internal":
            mask = ~snap.fixed & ~snap.roots | snap.unconstrained_roots
        compare_runs(ep_state.run_prior, snap, mask, **kwargs)

    @staticmethod
    def zero_posterior(snap):
        snap = snap.copy()
        snap.node_posterior[:] = 0.0
        snap.fields["node"][:] = 0.0
        return snap

    @staticmethod
    def negative_cavity_rate(snap):
        snap = snap.copy()
        free = snap.unconstrained_roots
        snap.fields["node"][free, MIXPRIOR, 1] = snap.node_posterior[free, 1] + 1.0
        return snap

    ERRORS = {
        # `penalty > 0` fails when the mean of shape / rate is infinite or negative:
        # free nodes whose cavity has zero rate (fixed leaves, or a root where every
        # update was skipped) or negative rate
        "all_nodes": NONPOSITIVE_PENALTY,
        "leaves": NONPOSITIVE_PENALTY,
        "zero_posterior": NONPOSITIVE_PENALTY,
        "negative_cavity_rate": NONPOSITIVE_PENALTY,
        # reachable from user input
        "max_shape_below_one": INVALID_MAX_SHAPE,
        "max_shape_zero": INVALID_MAX_SHAPE,
        "max_shape_nan": INVALID_MAX_SHAPE,
        # direct calls only
        "free_too_short": ValueError,
        "free_too_long": ValueError,
    }

    @pytest.mark.parametrize("case", ERRORS)
    def test_errors(self, snaps, case):
        snap = snaps["small_ts"]
        free = snap.unconstrained_roots
        kwargs = {}
        if case == "all_nodes":
            free = np.ones_like(free)
        elif case == "leaves":
            free = snap.leaves
        elif case == "zero_posterior":
            snap = self.zero_posterior(snap)
        elif case == "negative_cavity_rate":
            snap = self.negative_cavity_rate(snap)
        elif case.startswith("max_shape"):
            kwargs["max_shape"] = {
                "below_one": up(1.0, 0.0),
                "zero": 0.0,
                "nan": NAN,
            }[case[len("max_shape_") :]]
        elif case == "free_too_short":
            free = free[:-1]
        else:
            free = np.append(free, True)
        assert_run_kind(ep_state.run_prior, snap, self.ERRORS[case], free, **kwargs)

    def test_penalty_error_reachable_from_ep(self, fresh):
        # NOTE: current behaviour, possibly a bug: a likelihood pass in
        # which every update is skipped (a huge mutational target) leaves the roots
        # with a zero posterior, so that `iterate(regularise=True)` fails the
        # `penalty > 0` check (an error, not NaN).
        base = fresh["small_ts"].copy()
        base.edge_likelihoods = base.edge_likelihoods * [1.0, 1e300]

        def run(module):
            state = ep_state.run_likelihood(module, base, False)
            step = base.after_pass(state)
            return ep_state.run_prior(module, step, step.unconstrained_roots)

        result = compare(lambda: run(_tsdate), lambda: run(variational))
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
    def test_natural(self, snaps, ts_name, unphased):
        compare_runs(ep_state.run_mutations, snaps[ts_name], unphased)

    @pytest.mark.parametrize(
        ("ts_name", "unphased"),
        [
            ("small_ts", False),
            ("small_ts_historical", False),
            ("small_ts_unphased", True),
        ],
    )
    def test_forced_fixed_nodes(self, snaps, ts_name, unphased):
        # mutations on edges of every kind: both nodes fixed, one of the two fixed
        snap = ep_state.fixed_internal_snapshot(snaps[ts_name], unphased)
        counts = snap.branch_counts(unphased)
        assert all(counts[k] > 0 for k in ("both_fixed", "fixed_parent", "fixed_child"))
        compare_runs(ep_state.run_mutations, snap, unphased)

    @pytest.mark.parametrize("case", ["null_edges", "order_subset_and_repeats"])
    def test_mutation_selection(self, snaps, case):
        # mutations above the root have edge NULL and are skipped; the order is any
        # subset of the mutations, with repeats
        snap = snaps["small_ts"]
        kwargs = {}
        if case == "null_edges":
            edges = snap.mutation_edges.copy()
            edges[::7] = tskit.NULL
            kwargs["edges"] = edges
        else:
            kwargs["order"] = np.array([5, 3, 3, 100, 5], dtype=np.int32)
        compare_runs(ep_state.run_mutations, snap, False, **kwargs)

    @pytest.mark.parametrize(
        ("branch", "case"),
        [
            (b, c)
            for b in BRANCHES
            for c in ["valid", "projection_fails"]
            # fixed edges do not use the likelihood
            if not (c == "projection_fails" and "both_fixed" in b)
        ],
    )
    def test_single_edge(self, branch, case):
        # NOTE: current behaviour, possibly a bug: a mutation whose
        # projection fails gets NaN posterior and phase, silently left in the result.
        fails = case == "projection_fails"
        likelihood = (1.0, 1e300) if fails else (3.0, 0.1)
        snap = tiny_snapshot(branch, likelihood=likelihood)
        if fails and "sideways" in branch:
            # as in propagate_likelihood: C reports a division by zero in
            # `*_sideways_moments`
            with raises_zero_division():
                ep_state.run_mutations(_tsdate, snap, snap.unphased)
        else:
            compare_runs(ep_state.run_mutations, snap, snap.unphased)

    def test_twin_in_phased_pass(self):
        # unlike propagate_likelihood, there is no `unphased` check for p == c; the
        # twin projection is used as is (direct call only: edges never have p == c)
        snap = tiny_snapshot("twin")
        snap.mutation_blocks[:] = tskit.NULL
        assert_run_kind(ep_state.run_mutations, snap, "finite", False)

    @pytest.mark.parametrize(
        ("ages", "expected_kind"),
        [
            # the edge projection is uniform on (child, parent): variance 0 fails;
            # reachable with two samples of the same time, one the
            # ancestor of the other
            pytest.param([[5, 5], [5, 5]], "nan", id="equal_ages"),
            pytest.param([[5, 5], [up(5.0, 6.0)] * 2], "finite", id="nextafter_ages"),
            # NOTE: current behaviour, possibly a bug: ages in the wrong
            # order are not detected (the moments are symmetric)
            pytest.param([[10, 10], [0, 0]], "finite", id="inverted_ages"),
        ],
    )
    def test_fixed_edge_ages(self, ages, expected_kind):
        snap = tiny_snapshot("both_fixed")
        snap.node_constraints[:] = ages
        assert_run_kind(ep_state.run_mutations, snap, expected_kind, False)

    @pytest.mark.parametrize("change", ["edges_size", "posterior_size"])
    def test_shape_mismatches(self, snaps, change):
        # direct calls only: checked while parsing arguments in C
        snap = snaps["small_ts"]
        if change == "edges_size":
            kwargs = dict(edges=snap.mutation_edges[:-1])
        else:
            kwargs = dict(posterior=snap.node_posterior[:-1])
        assert_run_kind(ep_state.run_mutations, snap, ValueError, False, **kwargs)


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

    @pytest.mark.parametrize("case", CASES)
    def test_sweeps(self, fresh, case):
        name, kwargs = self.CASES[case]
        snap = fresh[name]
        sweeps = 4
        actual = ep_state.run_sweeps(_tsdate, snap, sweeps, **kwargs)
        expected = ep_state.run_sweeps(variational, snap, sweeps, **kwargs)
        for a, e in zip(actual[0], expected[0]):
            for x, y in zip(a, e):  # node posterior, edge and block lognorm
                np.testing.assert_array_equal(x, y)
        for x, y in zip(actual[1], expected[1]):  # mutation posterior, phase
            np.testing.assert_array_equal(x, y)
