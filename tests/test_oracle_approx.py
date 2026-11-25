"""
Parity of each implementation (``impl``) with the oracles in ``oracle/approx.py`` for
the gamma approximations: values to 4 ulp, NaN and infinity patterns, and which
exception is raised. Expected values always come from the oracle. Tables hold
deliberate edge cases (boundaries of the ``_valid_*`` checks, degenerate gamma fits,
regimes of the moment and projection functions); seeded random draws cover ranges
typical of expectation propagation (EP) and extreme ones.

Conventions of the C port pinned here: a NaN result (not an exception) signals a
parameter outside the valid region of the hypergeometric function; the ``*_projection``
functions then return NaN and, except for the ``mutation_*`` ones, their input
parameters unchanged; ``AssertionError`` signals a violated precondition (a fixed node
age that is not positive, or a Laplace root on the boundary); ``ZeroDivisionError``
signals float division by zero; ``KLMinimizationFailedError`` signals a failed gamma
fit. Accuracy of the approximations is measured in ``evaluation/numerical_accuracy.py``.
"""

import math

import numpy as np
import pytest
import scipy.special
from oracle import approx
from oracle.approx import KLMinimizationFailedError
from oracle.helpers import (
    INF,
    NAN,
    assert_outcome,
    compare,
    expected_error,
    kind,
    log_uniform,
    outcome,
)

nextafter = np.nextafter
MAX = np.finfo(np.float64).max
TINY = 5e-324


def parity(impl, name, args, expected_kind=None):
    """
    Compare ``impl.approx.<name>(*args)`` with the oracle (returning the oracle's
    outcome); optionally check its kind.
    """
    func = getattr(impl.approx, name)
    oracle = getattr(approx, name)
    if expected_kind is None:
        return compare(func, oracle, *args)
    return assert_outcome(func, oracle, args, expected_kind)


def table(name, rows):
    """``pytest.param`` rows of ``(name, args, expected_kind)`` with readable ids"""
    return [pytest.param(name, args, kind, id=f"{name}{args}") for args, kind in rows]


def floats(args):
    return tuple(float(a) for a in args)


def natural(shape, rate):
    """Natural parameters of a gamma"""
    return np.array([shape - 1.0, rate])


def gamma_quantiles(shape, rate, q1=0.25, q2=0.75):
    x1, x2 = scipy.special.gammaincinv(shape, [q1, q2]) / rate
    return q1, q2, float(x1), float(x2)


# --- validity checks --- #

MOMENTS_VALID = [
    ((1.0, 1.0), True),
    ((TINY, TINY), True),  # just inside
    ((MAX, MAX), True),
    ((0.0, 1.0), False),  # at the boundary
    ((1.0, 0.0), False),
    ((-0.0, 1.0), False),
    ((1.0, -0.0), False),
    ((nextafter(0.0, 1.0), 1.0), True),
    ((1.0, nextafter(0.0, 1.0)), True),
    ((nextafter(0.0, -1.0), 1.0), False),  # just outside
    ((1.0, nextafter(0.0, -1.0)), False),
    ((-1.0, 1.0), False),
    ((1.0, -1.0), False),
    ((INF, 1.0), False),
    ((1.0, INF), False),
    ((-INF, 1.0), False),
    ((NAN, 1.0), False),
    ((1.0, NAN), False),
]

HYP1F1 = [
    ((1.0, 1.0, 0.0), True),  # b == a is allowed
    ((1.0, nextafter(1.0, 0.0), 0.0), False),  # b just below a
    ((1.0, nextafter(1.0, 2.0), 0.0), True),
    ((TINY, 1.0, 0.0), True),  # a just above zero
    ((0.0, 1.0, 0.0), False),
    ((-0.0, 1.0, 0.0), False),
    ((nextafter(0.0, -1.0), 1.0, 0.0), False),
    ((1.0, 2.0, 1e300), True),  # z may have any sign and size
    ((1.0, 2.0, -1e300), True),
    ((1.0, 2.0, TINY), True),
    ((1.0, 2.0, INF), False),
    ((1.0, 2.0, -INF), False),
    ((1.0, 2.0, NAN), False),
    ((NAN, 2.0, 0.0), False),
    ((1.0, NAN, 0.0), False),
    ((INF, INF, 0.0), False),
    ((1.0, INF, 0.0), False),
    ((MAX, MAX, 0.0), True),
]

HYPERU = [
    ((1.0, 1.0, 1.0), False),  # b == a is not allowed here (unlike hyp1f1)
    ((1.0, nextafter(1.0, 2.0), 1.0), True),
    ((1.0, nextafter(1.0, 0.0), 1.0), False),
    ((TINY, 1.0, 1.0), True),
    ((0.0, 1.0, 1.0), False),
    ((-0.0, 1.0, 1.0), False),
    ((1.0, 2.0, 0.0), False),  # z == 0
    ((1.0, 2.0, -0.0), False),
    ((1.0, 2.0, TINY), True),  # z just above zero
    ((1.0, 2.0, nextafter(0.0, -1.0)), False),
    ((1.0, 2.0, -1.0), False),
    ((1.0, 2.0, 1e300), True),
    ((1.0, 2.0, INF), False),
    ((1.0, 2.0, NAN), False),
    ((NAN, 2.0, 1.0), False),
    ((1.0, INF, 1.0), False),
    ((INF, INF, 1.0), False),
]


def z_below_one(z):
    """The condition on z: below one, or for z < 0 that ``z / (z - 1)`` is below one"""
    return bool(z / (z - 1) < 1.0 if z < 0.0 else z < 1.0)


def hyp2f1_rows():
    """``(args, expected)`` for ``_valid_hyp2f1``: a, b, c positive and finite, z < 1"""
    rows = [((1.0, 1.0, 1.0, 0.0), True), ((10.0, 1.0, 1.0, 0.5), True)]  # a > c is ok
    for i in range(3):
        for z in [0.5, -0.0]:
            for bad, valid in [(0.0, False), (-0.0, False), (TINY, True)]:
                abc = [1.0, 1.0, 1.0]
                abc[i] = bad
                rows.append(((*abc, z), valid))
        for bad in [INF, NAN, -TINY]:
            abc = [1.0, 1.0, 1.0]
            abc[i] = bad
            rows.append(((*abc, 0.5), False))
    zs = [0.0, -0.0, TINY, 0.5, 1 - 2.0**-53, 1.0, 1 + 2.0**-52, 2.0]
    zs += [NAN, INF, -INF, -1.0, -1e15]
    rows += [((1.0, 2.0, 3.0, z), z_below_one(z)) for z in zs]
    # NOTE: current behaviour, possibly a bug: for z < 0 the check
    # z / (z - 1) < 1 fails once z - 1 rounds to z, and between -2**53 and -2**54 it
    # alternates with the last bit of z
    big = 2.0**53
    zs = [-(big - 2), -(big - 1), -big, -(big + 2), -(big + 4), -(big + 6), -9e15]
    zs += [-(2 * big - 2), -2 * big, -(2 * big + 4), -1e17, -1e300, -MAX]
    return rows + [((1.0, 2.0, 3.0, z), z_below_one(z)) for z in zs]


VALID = (
    [
        (n, args, v)
        for n in ["_valid_moments", "_valid_gamma"]
        for args, v in MOMENTS_VALID
    ]
    + [("_valid_hyp1f1", args, v) for args, v in HYP1F1]
    + [("_valid_hyperu", args, v) for args, v in HYPERU]
    + [("_valid_hyp2f1", args, v) for args, v in hyp2f1_rows()]
)


class TestValidChecks:
    """
    Boundaries of the ``_valid_*`` checks, at and either side. Each of them is the
    gate that turns unusable moments or hypergeometric parameters into NaN.
    """

    @pytest.mark.parametrize(
        ("name", "args", "expected"),
        [pytest.param(*row, id=f"{row[0]}{row[1]}") for row in VALID],
    )
    def test_valid(self, impl, name, args, expected):
        result = parity(impl, name, args)
        assert bool(result.value) is expected


# --- gamma approximations --- #

LOG_MOMENTS = [
    ((1.0, 1.0), "finite"),
    ((1e300, 1e300), "finite"),
    ((1e-100, 1e-100), "finite"),
    ((0.0, 1.0), AssertionError),
    ((1.0, 0.0), AssertionError),
    ((-1.0, 1.0), AssertionError),
    ((1.0, -1.0), AssertionError),
    ((NAN, 1.0), AssertionError),
    ((1.0, NAN), AssertionError),
    ((INF, 1.0), "nan"),
    ((1.0, INF), "nonfinite"),
    ((1e-300, 1e-300), ZeroDivisionError),  # mean**2 underflows
    ((TINY, 1.0), ZeroDivisionError),
    ((1e-200, 1.0), ZeroDivisionError),
]

GAMMA_KL = [
    # (x, logx, expected kind); `gap` = log(x) - logx must be positive (Jensen)
    ((1.0, 0.0), KLMinimizationFailedError),  # gap == 0
    ((1.0, nextafter(0.0, -1.0)), "nonfinite"),  # smallest gap: alpha, rate overflow
    ((1.0, -1e-300), "finite"),
    ((1.0, -1e-17), "finite"),
    ((1.0, -1e-10), "finite"),
    ((1.0, -4e-5), "finite"),
    ((1.0, -0.5), "finite"),
    ((1.0, -1.0), "finite"),  # shape < 1
    ((1.0, -10.0), "finite"),
    ((1.0, -1e10), "finite"),
    ((1.0, -1e160), "finite"),
    ((1.0, -1e164), ZeroDivisionError),  # shape bound ~1e-164: trigamma divides by 0
    ((1.0, -1e300), ZeroDivisionError),
    ((1.0, -INF), KLMinimizationFailedError),
    ((1.0, INF), KLMinimizationFailedError),
    ((1.0, NAN), KLMinimizationFailedError),  # not (log x > nan)
    ((0.0, -1.0), KLMinimizationFailedError),
    ((-1.0, -1.0), KLMinimizationFailedError),
    ((-0.0, -1.0), KLMinimizationFailedError),
    ((NAN, -1.0), KLMinimizationFailedError),
    ((INF, -1.0), ZeroDivisionError),  # alpha = 0.5 / inf = 0
    ((INF, INF), KLMinimizationFailedError),
    ((1e-300, -1.0), KLMinimizationFailedError),  # log x = -690 < logx
    ((1e-300, -700.0), "finite"),
    ((TINY, -745.0), "nonfinite"),  # rate = alpha / x overflows
    ((1e300, 1.0), "finite"),
    ((1e300, 690.0), "finite"),
    ((1e308, 709.0), "finite"),
    ((2.0, 0.5), "finite"),
    ((100.0, math.log(100.0) - 1.0), "finite"),
    ((1e5, math.log(1e5) - 0.5e-4), "finite"),
    ((1e5, math.log(1e5) - 0.5e-8), "finite"),
]

GAMMA_MOM = [
    ((1.0, 1.0), "finite"),
    ((1.0, 0.0), KLMinimizationFailedError),
    ((0.0, 1.0), KLMinimizationFailedError),
    ((-0.0, 1.0), KLMinimizationFailedError),
    ((-1.0, 1.0), KLMinimizationFailedError),
    ((1.0, -1.0), KLMinimizationFailedError),
    ((NAN, 1.0), KLMinimizationFailedError),
    ((1.0, NAN), KLMinimizationFailedError),
    ((TINY, TINY), "finite"),
    ((1.0, 1e-300), "finite"),  # shape 1e300
    ((1.0, TINY), "nonfinite"),  # shape overflows
    ((INF, 1.0), "nonfinite"),
    ((1.0, INF), "finite"),  # shape 0 - 1
    ((INF, INF), "nan"),
    # NOTE: current behaviour, possibly a bug: mean**2 underflows, shape -1
    ((1e-200, 1e-200), "finite"),
    ((1e200, 1e200), "nonfinite"),  # mean**2 overflows
    ((1e154, 1.0), "finite"),
    ((1.0, 1e300), "finite"),
    ((1e8, 1.0), "finite"),
    ((1e3, 1e-6), "finite"),
]


GAMMA_IQR_ERRORS = [
    ((0.25, 0.75, 2.0, 1.0, 50.0), KLMinimizationFailedError),  # x unsorted
    ((0.75, 0.25, 1.0, 2.0, 50.0), KLMinimizationFailedError),  # q unsorted
    ((0.25, 0.25, 1.0, 2.0, 50.0), KLMinimizationFailedError),  # q equal
    ((0.25, 0.75, 1.0, NAN, 50.0), KLMinimizationFailedError),
    ((NAN, 0.75, 1.0, 2.0, 50.0), KLMinimizationFailedError),
    ((0.25, 0.75, 0.0, 1.0, 50.0), ZeroDivisionError),  # log(x2 / 0)
    ((0.0, 0.75, 1.0, 2.0, 50.0), ZeroDivisionError),  # log(q2 / 0)
    ((0.25, 0.75, -1.0, 1.0, 50.0), KLMinimizationFailedError),
    ((0.25, 1.0, 1.0, 2.0, 50.0), KLMinimizationFailedError),  # q2 == 1: negative shape
    ((0.25, 0.75, 1.0, INF, 50.0), AssertionError),
    ((0.25, 0.75, 0.0, 0.0, 50.0), ZeroDivisionError),  # x1 == x2 == 0
    # the shape is capped at max_shape when x1 == x2, when the lower bound exceeds it
    # or when Newton's iteration ends above it
    ((0.25, 0.75, 3.0, 3.0, 50.0), "finite"),
    ((0.25, 0.75, 1.0, 1.000001, 1e5), "finite"),
    # the gradient vanishes with a high cap: the update divides by zero
    ((0.25, 0.75, 1.0, 1.0001, 1e6), ZeroDivisionError),
    # NOTE: current behaviour, possibly a bug: max_shape is not validated
    ((*gamma_quantiles(2.0, 1.0), 0.0), "nan"),
    ((*gamma_quantiles(2.0, 1.0), -1.0), "nan"),
    ((*gamma_quantiles(2.0, 1.0), 1.0), "finite"),
    ((*gamma_quantiles(2.0, 1.0), INF), "finite"),
    # NOTE: current behaviour, possibly a bug: the lower quantile of a
    # gamma with tiny shape underflows to zero, so the fit divides by zero
    ((*gamma_quantiles(1e-3, 1.0), 50.0), ZeroDivisionError),
]
# shapes below, at and above the cap, and beyond the range where `_gammainc_der` uses
# its gaussian approximation (shape > 1e3)
GAMMA_IQR_FITS = [
    (shape, max_shape)
    for shape in [0.01, 0.5, 1.0, 10.0, 49.0, 50.0, 51.0, 999.0, 1001.0, 1e4, 1e8]
    for max_shape in [10.0, 50.0, 1000.0, 1e5]
]

AVERAGE_GAMMAS = {
    "single": ([1.0], [1.0], "finite"),
    "pair": ([0.5, 1.5], [1.0, 1.0], "finite"),
    "identical": ([1.0, 1.0], [1.0, 1.0], "finite"),
    "tiny shape": ([1e-10], [1.0], "finite"),
    "huge shapes": ([1e8, 1e8], [1.0, 1e-3], "finite"),
    "mixed shapes": ([0.5, 1e6], [1.0, 1e-3], "finite"),
    "wide rates": ([1.0, 2.0], [1e-300, 1e300], "finite"),
    "shape+1 near zero": ([-1.0 + 1e-10, 4.0], [1.0, 2.0], "finite"),
    "shape+1 zero": ([-1.0], [1.0], ZeroDivisionError),  # digamma pole
    "shape+1 negative": ([-2.0], [1.0], KLMinimizationFailedError),
    "zero rate": ([1.0], [0.0], ZeroDivisionError),
    "negative rate": ([1.0], [-1.0], KLMinimizationFailedError),
    "nan shape": ([NAN], [1.0], KLMinimizationFailedError),
    "inf shape": ([INF], [1.0], KLMinimizationFailedError),
    "empty": ([], [], ZeroDivisionError),
    "size mismatch": ([1.0], [1.0, 2.0], AssertionError),
}

GAMMA_DOMAIN = (
    table("approximate_log_moments", LOG_MOMENTS)
    + table("approximate_gamma_kl", GAMMA_KL)
    + table("approximate_gamma_mom", GAMMA_MOM)
    + table("approximate_gamma_iqr", GAMMA_IQR_ERRORS)
)


class TestGammaApproximations:
    @pytest.mark.parametrize(("name", "args", "expected"), GAMMA_DOMAIN)
    def test_domain(self, impl, name, args, expected):
        parity(impl, name, args, expected)

    @pytest.mark.parametrize(
        ("name", "args", "expected"),
        [p for p in GAMMA_DOMAIN if p.values[2] is KLMinimizationFailedError],
    )
    def test_error_messages(self, impl, name, args, expected):
        # the messages distinguish the failure modes; the port re-raises them
        func = getattr(impl.approx, name)
        with pytest.raises(expected_error(KLMinimizationFailedError)):
            func(*args)
        compare(func, getattr(approx, name), *args)

    @pytest.mark.parametrize("gap", [1e-17, 1e-10, 4e-5, 4.99e-5, 5e-5, 6e-5, 1e-3])
    def test_kl_asymptotic_threshold(self, impl, gap):
        # for alpha > 1e4 (gap < 5e-5) the lower bound alpha = 0.5 / gap is returned
        # without iterating; above, Newton's iteration refines it
        result = parity(impl, "approximate_gamma_kl", (1.0, -gap), "finite")
        shape = result.value[0]
        if gap < 5e-5:
            assert shape + 1.0 == 0.5 / gap
        else:
            assert shape + 1.0 != 0.5 / gap

    @pytest.mark.parametrize(("shape", "max_shape"), GAMMA_IQR_FITS, ids=repr)
    def test_iqr_fits(self, impl, shape, max_shape):
        args = (*gamma_quantiles(shape, 0.001), max_shape)
        parity(impl, "approximate_gamma_iqr", args, "finite")


class TestAverageGammas:
    @pytest.mark.parametrize("label", list(AVERAGE_GAMMAS))
    def test_cases(self, impl, label):
        shapes, rates, expected = AVERAGE_GAMMAS[label]
        args = np.array(shapes, dtype=float), np.array(rates, dtype=float)
        parity(impl, "average_gammas", args, expected)


# --- moments and projections --- #

# regimes for functions of two free nodes: (a_i, b_i, a_j, b_j, y, mu), with the gamma
# shapes a_i, a_j, rates b_i, b_j, mutation count y and rate mu. With t = mu + b_i and
# z = (mu - b_j) / t the hypergeometric argument is z (parent above child) or 1 - z
# (unphased).
TWO_NODE = {
    "typical": (10, 0.01, 10, 0.01, 1, 1e-4),
    "y=0": (10, 0.01, 10, 0.01, 0, 1e-4),
    "y=100": (10, 0.01, 10, 0.01, 100, 1e-3),
    "y=1e4": (10, 0.01, 10, 0.01, 1e4, 1e-3),
    "mu=1e-12": (10, 0.01, 10, 0.01, 1, 1e-12),
    "mu=0": (10, 0.01, 10, 0.01, 0, 0.0),
    "mu=1e6": (10, 0.01, 10, 0.01, 1, 1e6),
    "shapes=1e4": (1e4, 10, 1e4, 10, 1, 1e-4),
    # NOTE: current behaviour, possibly a bug: the variance is a difference
    # of the second moment and the squared mean, and goes negative at huge shapes
    "shapes=1e8": (1e8, 1e5, 1e8, 1e5, 1, 1e-4),
    "shapes=1e-3": (1e-3, 1e-5, 1e-3, 1e-5, 1, 1e-4),
    "rates=1e-12": (10, 1e-12, 10, 1e-12, 1, 1e-14),
    "rates=1e8": (10, 1e8, 10, 1e8, 1, 1e5),
    "t=0": (10, -1e-4, 10, 0.01, 1, 1e-4),
    "t<0": (10, -1e-3, 10, 0.01, 1, 1e-4),
    "t=1e-20": (10, nextafter(-1e-4, 0.0), 10, 0.01, 1, 1e-4),
    "z=1": (10, 0.01, 10, -0.01, 1, 1e-4),
    "z=1-1e-10": (10, 0.01, 10, -0.01 + 1e-12, 1, 1e-4),
    "z>1": (10, 0.01, 10, -0.02, 1, 1e-4),
    "z=0": (10, 0.01, 10, 1e-4, 1, 1e-4),
    "z=-1e9": (10, 0.01, 10, 1e9, 1, 1e-4),
    "z=-1e5": (10, 0.01, 10, 1e3, 1, 1e-4),
    "a_j=0": (10, 0.01, 0, 0.01, 1, 1e-4),
    "a_j=1e-8": (10, 0.01, 1e-8, 0.01, 1, 1e-4),
    "a_i=1e8, a_j=1": (1e8, 0.01, 1, 0.01, 1, 1e-4),
    "b_j<0 (damped)": (10, 0.01, 10, -0.005, 1, 1e-4),
    "b_j=inf": (10, 0.01, 10, INF, 1, 1e-4),
}

# regimes for a fixed node and one free node: (t, a, b, y, mu) with the free node's
# shape a, rate b; t is the fixed age (child for rootward, parent otherwise).
ONE_NODE = {
    "typical": (1000, 10, 0.01, 1, 1e-4),
    "y=0": (1000, 10, 0.01, 0, 1e-4),
    "y=1e4": (1000, 10, 0.01, 1e4, 1e-3),
    "t=1e-10": (1e-10, 10, 0.01, 1, 1e-4),
    "t=5e-324": (TINY, 10, 0.01, 1, 1e-4),
    "t=1e10": (1e10, 10, 0.01, 1, 1e-4),
    "t=0": (0.0, 10, 0.01, 1, 1e-4),
    "t=-0": (-0.0, 10, 0.01, 1, 1e-4),
    "t<0": (-1.0, 10, 0.01, 1, 1e-4),
    "t=nan": (NAN, 10, 0.01, 1, 1e-4),
    "t=inf": (INF, 10, 0.01, 1, 1e-4),
    "mu=1e-14": (1000, 10, 0.01, 1, 1e-14),
    "mu=1e5": (1000, 10, 0.01, 1, 1e5),
    "mu=0": (1000, 10, 0.01, 0, 0.0),
    "shapes=1e4": (1000, 1e4, 10, 1, 1e-4),
    "shapes=1e8": (1000, 1e8, 1e5, 1, 1e-4),
    "shapes=1e-3": (1000, 1e-3, 1e-5, 1, 1e-4),
    "rates=1e-12": (1000, 10, 1e-12, 1, 1e-14),
    "rates=1e8": (1e-5, 10, 1e8, 1, 1e5),
    "rate+mu=0": (1000, 10, -1e-4, 1, 1e-4),
    "rate+mu<0": (1000, 10, -1e-3, 1, 1e-4),
    "rate<0, rate+mu>0": (1000, 10, -5e-5, 1, 1e-4),
    "a=0": (1000, 0, 0.01, 1, 1e-4),
    "a<0": (1000, -1, 0.01, 1, 1e-4),
    # NOTE: current behaviour, possibly a bug: y = -1 passes
    # `_valid_hyp1f1` (b == a) but `_hyp1f1_laplace` asserts b > a
    "y=-1": (1000, 10, 0.01, -1, 1e-4),
    "y=-0.5": (1000, 10, 0.01, -0.5, 1e-4),
}

# (a_i, b_i, y, mu): a node above both edges of an unphased individual
TWIN = {
    "typical": (10, 0.01, 1, 1e-4),
    "y=0": (10, 0.01, 0, 1e-4),
    "mu=0": (10, 0.01, 0, 0.0),
    "shapes=1e8": (1e8, 1e5, 1, 1e-4),
    "shapes=1e-3": (1e-3, 1e-5, 1, 1e-4),
    "rates=1e-12": (10, 1e-12, 1, 1e-14),
    "rates=1e8": (10, 1e8, 1, 1e5),
    "r=0": (10, -2e-4, 1, 1e-4),  # r = b_i + 2 mu
    "r<0": (10, -1e-3, 1, 1e-4),
    "r=1e-300": (10, 1e-300, 1, 0.0),
    "s=0": (0, 0.01, 0, 1e-4),  # s = a_i + y
    "s<0": (-1, 0.01, 0, 1e-4),
    "a_i=nan": (NAN, 0.01, 1, 1e-4),
    "a_i=inf": (INF, 0.01, 1, 1e-4),
    "y=1e300": (10, 0.01, 1e300, 1e-4),
    "mu=1e300": (10, 0.01, 1, 1e300),
}

# (t_i, t_j): fixed parent and child ages
TIMES = {
    "3,1": (3, 1),
    "1,3 (child older)": (1, 3),
    # NOTE: current behaviour, possibly a bug: equal ages give zero variance, so the
    # edge projection is NaN, but inverted ages (1,3) give a valid result
    "equal": (1, 1),
    "0,0": (0, 0),
    "0,1": (0, 1),
    "1,0": (1, 0),
    "-0,1": (-0.0, 1),
    # NOTE: current behaviour, possibly a bug: edge moments do not check the ages
    "neg,1": (-1, 1),
    "1,neg": (1, -1),
    "1e200,1e-200": (1e200, 1e-200),
    "1e300,1": (1e300, 1),
    "1e308,1e307": (1e308, 1e307),
    "1e-200,1e-210": (1e-200, 1e-210),
    "1e-160,1e-170": (1e-160, 1e-170),
    "5e-324,5e-324": (TINY, TINY),
    "1,1-eps": (1.0, nextafter(1.0, 0.0)),
    "nan,1": (NAN, 1),
    "inf,1": (INF, 1),
    "inf,inf": (INF, INF),
}


TWO_NODE_MOMENTS = ["moments", "unphased_moments", "mutation_moments"]
TWO_NODE_MOMENTS += ["mutation_unphased_moments"]
ONE_NODE_MOMENTS = ["rootward_moments", "mutation_rootward_moments", "leafward_moments"]
ONE_NODE_MOMENTS += ["mutation_leafward_moments", "sideways_moments"]
ONE_NODE_MOMENTS += ["mutation_sideways_moments"]
TWIN_MOMENTS = ["twin_moments", "mutation_twin_moments"]
TIMES_MOMENTS = ["mutation_edge_moments", "mutation_block_moments"]

# the regimes (inputs of the moment function) in which each moment function is called
REGIMES = {
    **{name: TWO_NODE for name in TWO_NODE_MOMENTS},
    **{name: ONE_NODE for name in ONE_NODE_MOMENTS},
    **{name: TWIN for name in TWIN_MOMENTS},
    **{name: TIMES for name in TIMES_MOMENTS},
}

# For each projection: the moment function it wraps and the checks it applies to the
# moments before fitting a gamma. `pairs` indexes (mean, variance) in the result of
# the moment function; `prob` indexes a probability that must lie in [0, 1].
PROJECTIONS = {
    "gamma_projection": ("moments", [(1, 2), (3, 4)], None),
    "unphased_projection": ("unphased_moments", [(1, 2), (3, 4)], None),
    "rootward_projection": ("rootward_moments", [(1, 2)], None),
    "leafward_projection": ("leafward_moments", [(1, 2)], None),
    "sideways_projection": ("sideways_moments", [(1, 2)], None),
    "twin_projection": ("twin_moments", [(1, 2)], None),
    "mutation_gamma_projection": ("mutation_moments", [(0, 1)], None),
    "mutation_rootward_projection": ("mutation_rootward_moments", [(0, 1)], None),
    "mutation_leafward_projection": ("mutation_leafward_moments", [(0, 1)], None),
    "mutation_edge_projection": ("mutation_edge_moments", [(0, 1)], None),
    "mutation_unphased_projection": ("mutation_unphased_moments", [(1, 2)], 0),
    "mutation_sideways_projection": ("mutation_sideways_moments", [(1, 2)], 0),
    "mutation_twin_projection": ("mutation_twin_moments", [(1, 2)], 0),
    "mutation_block_projection": ("mutation_block_moments", [(1, 2)], 0),
}
# projections that return their input parameters unchanged when the moments are invalid
RETURNS_INPUT = {
    "gamma_projection",
    "unphased_projection",
    "rootward_projection",
    "leafward_projection",
    "sideways_projection",
    "twin_projection",
}


def projection_args(name, args):
    """
    Arguments of a projection from the arguments of the moment function it wraps
    (shapes, not natural parameters).
    """
    moments = PROJECTIONS[name][0]
    if moments in TWO_NODE_MOMENTS:
        a_i, b_i, a_j, b_j, y, mu = args
        return natural(a_i, b_i), natural(a_j, b_j), np.array([y, mu])
    if moments in TWIN_MOMENTS:
        a_i, b_i, y, mu = args
        return natural(a_i, b_i), np.array([y, mu])
    if moments in TIMES_MOMENTS:
        return tuple(args)
    t, a, b, y, mu = args
    return t, natural(a, b), np.array([y, mu])


def moment_args(name, proj_args):
    """
    The arguments the projection passes to its moment function: shapes are recovered
    from the natural parameters, so tiny shapes differ from the original by rounding.
    """
    moments = PROJECTIONS[name][0]
    if moments in TWO_NODE_MOMENTS:
        pars_i, pars_j, (y, mu) = proj_args
        return pars_i[0] + 1, pars_i[1], pars_j[0] + 1, pars_j[1], y, mu
    if moments in TWIN_MOMENTS:
        pars_i, (y, mu) = proj_args
        return pars_i[0] + 1, pars_i[1], y, mu
    if moments in TIMES_MOMENTS:
        return proj_args
    t, pars, (y, mu) = proj_args
    return t, pars[0] + 1, pars[1], y, mu


def is_valid(mean, variance, prob=None):
    if prob is not None and not (0 <= prob <= 1):
        return False
    return bool(np.isfinite(mean) and np.isfinite(variance) and mean > 0 and variance > 0)


def check_projection(impl, name, args):
    """
    Compare a projection with the oracle, and check the contract on the oracle's
    result: the projection raises as its moment function does; otherwise it is NaN
    exactly when the moments are unusable (non-finite, non-positive, or a probability
    outside [0, 1]), and on NaN the non-mutation projections return the input
    parameters unchanged and the mutation ones NaN.
    """
    moments, pairs, prob_index = PROJECTIONS[name]
    proj_args = projection_args(name, args)
    result = parity(impl, name, proj_args)
    moment_result = outcome(getattr(approx, moments), *moment_args(name, proj_args))
    if moment_result.exception is not None:
        assert type(result.exception) is type(moment_result.exception)
        return
    values = moment_result.value
    prob = values[prob_index] if prob_index is not None else None
    expected_nan = not all(is_valid(values[i], values[j], prob) for i, j in pairs)
    assert kind(result) != "nan" or expected_nan
    if not expected_nan:
        assert not np.isnan(result.value[0])
        return
    assert np.isnan(result.value[0])
    if name in RETURNS_INPUT:
        given = [p for p in proj_args if isinstance(p, np.ndarray)]
        for returned, original in zip(result.value[1:], given):
            np.testing.assert_array_equal(returned, original)
    else:
        assert np.isnan(result.value[1]).all()


MOMENT_REGIMES = [
    pytest.param(name, label, id=f"{name}-{label}")
    for name, regimes in REGIMES.items()
    for label in regimes
]
PROJECTION_REGIMES = [
    pytest.param(name, label, id=f"{name}-{label}")
    for name, (moments, _, _) in PROJECTIONS.items()
    for label in REGIMES[moments]
]


class TestMoments:
    @pytest.mark.parametrize(("name", "label"), MOMENT_REGIMES)
    def test_regimes(self, impl, name, label):
        parity(impl, name, floats(REGIMES[name][label]))


class TestProjections:
    @pytest.mark.parametrize(("name", "label"), PROJECTION_REGIMES)
    def test_regimes(self, impl, name, label):
        moments = PROJECTIONS[name][0]
        check_projection(impl, name, floats(REGIMES[moments][label]))

    @pytest.mark.parametrize("name", list(PROJECTIONS))
    def test_inputs_are_not_modified(self, impl, name):
        moments = PROJECTIONS[name][0]
        first_regime = next(iter(REGIMES[moments].values()))
        proj_args = projection_args(name, floats(first_regime))
        copies = [a.copy() if isinstance(a, np.ndarray) else a for a in proj_args]
        getattr(impl.approx, name)(*proj_args)
        for arg, copy in zip(proj_args, copies):
            np.testing.assert_array_equal(arg, copy)

    def test_unphased_probability_out_of_range(self, impl):
        # NOTE: current behaviour, possibly a bug: the Laplace
        # approximation can give a probability above one for finite mean and variance,
        # which the projection turns into NaN
        args = (
            1289.8261024396659,
            2.1161296052067344,
            0.15552864791315738,
            0.0007984796762152916,
            1.0,
            4.811169071413665e-07,
        )
        parity(impl, "mutation_unphased_moments", args, "finite")
        name = "mutation_unphased_projection"
        parity(impl, name, projection_args(name, args), "nan")


# --- seeded random draws --- #


def flip_sign(rng, values, prob=0.12):
    """Negate one of ``values`` with probability ``prob``, to give invalid input."""
    values = list(values)
    if rng.random() < prob:
        i = rng.integers(len(values))
        values[i] = -values[i]
    return values


def draw_gamma(rng):
    """Shape and rate of a node age"""
    return float(rng.uniform(0.5, 30)), log_uniform(rng, -5, -2)


def draw_edge(rng):
    """Mutation count and mutational target size on an edge"""
    return float(rng.integers(0, 12)), log_uniform(rng, -4, -2)


def draw_two_node(rng, extreme):
    if extreme:
        a_i, b_i, a_j, b_j = (log_uniform(rng, -8, 8) for _ in range(4))
        b_i *= float(rng.choice([-1.0, 1.0, 1.0, 1.0]))
        b_j *= float(rng.choice([-1.0, 1.0, 1.0, 1.0]))
        y = float(rng.choice([0, 0, 1, 2, 5, 100, 1e4]))
        return a_i, b_i, a_j, b_j, y, log_uniform(rng, -12, 3)
    a_i, b_i = draw_gamma(rng)
    a_j, b_j = draw_gamma(rng)
    y, mu = draw_edge(rng)
    return (*flip_sign(rng, [a_i, b_i, a_j, b_j]), y, mu)


def draw_one_node(rng, extreme):
    t = 0.0 if rng.random() < (0.15 if extreme else 0.1) else None
    if extreme:
        t = log_uniform(rng, -6, 8) if t is None else t
        a, b = log_uniform(rng, -8, 8), log_uniform(rng, -8, 8)
        b *= float(rng.choice([-1.0, 1.0, 1.0, 1.0]))
        y = float(rng.choice([0, 0, 1, 2, 5, 100, 1e4]))
        return t, a, b, y, log_uniform(rng, -12, 3)
    t = log_uniform(rng, 0, 4) if t is None else t
    a, b = flip_sign(rng, draw_gamma(rng))
    return (t, a, b, *draw_edge(rng))


def draw_twin(rng, extreme):
    if extreme:
        a_i, b_i = log_uniform(rng, -8, 8), log_uniform(rng, -8, 8)
        b_i *= float(rng.choice([-1.0, 1.0, 1.0]))
        y = float(rng.choice([0, 1, 5, 1e6]))
        return a_i, b_i, y, log_uniform(rng, -12, 3)
    a_i, b_i = flip_sign(rng, draw_gamma(rng))
    return (a_i, b_i, *draw_edge(rng))


def draw_times(rng, extreme):
    lo, hi = (-6, 8) if extreme else (0, 4)
    return log_uniform(rng, lo, hi), log_uniform(rng, lo, hi)


DRAW_GEOMETRY = {
    **{name: draw_two_node for name in TWO_NODE_MOMENTS},
    **{name: draw_one_node for name in ONE_NODE_MOMENTS},
    **{name: draw_twin for name in TWIN_MOMENTS},
    **{name: draw_times for name in TIMES_MOMENTS},
}


def draw_log_moments(rng):
    mean = log_uniform(rng, -1, 4)
    return mean, mean**2 * log_uniform(rng, -3, 0)


def draw_gamma_kl(rng):
    # log E[x] > E[log x] by Jensen; sometimes violated or nonpositive
    x = log_uniform(rng, -2, 4)
    gap = log_uniform(rng, -4, 1)
    if rng.random() < 0.1:
        gap = -gap
    return x, float(np.log(x) - gap)


def draw_gamma_mom(rng):
    mean = log_uniform(rng, -1, 4)
    variance = mean**2 * log_uniform(rng, -3, 1)
    return tuple(flip_sign(rng, [mean, variance], 0.1))


def draw_gamma_iqr(rng):
    # empirical quantiles of a gamma, so that the shape is recoverable
    q1, q2, x1, x2 = gamma_quantiles(
        log_uniform(rng, -0.5, 2.5), log_uniform(rng, -5, -2)
    )
    max_shape = float(rng.choice([50.0, 1000.0]))
    u = rng.random()
    if u < 0.05:
        x2 = x1  # degenerate, capped shape
    elif u < 0.1:
        x1, x2 = x2, x1  # unsorted
    return q1, q2, x1, x2, max_shape


def draw_average_gammas(rng):
    n = int(rng.integers(1, 8))
    return rng.uniform(0.5, 30, n) - 1, 10.0 ** rng.uniform(-5, -2, n)


MODES = ["typical", "extreme"]
DRAW_HELPERS = {
    "approximate_log_moments": draw_log_moments,
    "approximate_gamma_kl": draw_gamma_kl,
    "approximate_gamma_mom": draw_gamma_mom,
    "approximate_gamma_iqr": draw_gamma_iqr,
    "average_gammas": draw_average_gammas,
}

DRAWS = [(name, "typical") for name in DRAW_HELPERS]
DRAWS += [(name, mode) for name in [*REGIMES, *PROJECTIONS] for mode in MODES]


class TestRandomDraws:
    """
    Seeded draws in ranges typical of EP on tree sequences (gamma shapes of order
    1-30, rates of order 1/(node age), mutation counts 0-12 per edge, a fraction with a
    sign flipped so that validity checks fail) and over 16 orders of magnitude
    ("extreme", where exceptions, NaN and infinity are common). The failing draw is
    named in the assertion.
    """

    @pytest.mark.parametrize("seed", [1, 2, 3])
    @pytest.mark.parametrize(("name", "mode"), DRAWS)
    def test_random_draws(self, impl, name, mode, seed):
        rng = np.random.default_rng(seed)
        extreme = mode == "extreme"
        for i in range(30):
            if name in DRAW_HELPERS:
                args = DRAW_HELPERS[name](rng)
                check = parity
            elif name in PROJECTIONS:
                args = DRAW_GEOMETRY[PROJECTIONS[name][0]](rng, extreme)
                check = check_projection
            else:
                args = DRAW_GEOMETRY[name](rng, extreme)
                check = parity
            try:
                check(impl, name, args)
            except AssertionError as e:
                raise AssertionError(f"draw {i}: {name}{args}: {e}") from None
