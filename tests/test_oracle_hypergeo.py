"""
Parity of the C implementation (``impl``) with the oracles in ``oracle/hypergeo.py``
for the special functions: exact values (including NaNs and infinities), and which
exception is raised (``AssertionError`` for the domain checks,
``ZeroDivisionError`` for float division by zero). Expected values always come from the
oracle. Tables hold deliberate edge cases (poles, domain boundaries, both sides of a
branch switch); grids and seeded random draws cover the interior and extreme ranges.
Accuracy of the approximations is measured in ``evaluation/numerical_accuracy.py``.
"""

import numpy as np
import pytest
from oracle import hypergeo
from oracle._libm import lgamma, log
from oracle.helpers import (
    INF,
    NAN,
    assert_outcome,
    assert_same_values,
    compare,
    log_uniform,
)

nextafter = np.nextafter


def parity(impl, name, args, expected_kind=None):
    """
    Compare ``impl.hypergeo.<name>(*args)`` with the oracle; optionally check kind.
    """
    func = getattr(impl.hypergeo, name)
    oracle = getattr(hypergeo, name)
    if expected_kind is None:
        return compare(func, oracle, *args)
    return assert_outcome(func, oracle, args, expected_kind)


def table(name, rows):
    """
    ``pytest.param`` rows of ``(name, args, expected_kind)`` with readable ids
    """
    return [pytest.param(name, args, kind, id=f"{name}{args}") for args, kind in rows]


DIGAMMA_POINTS = [
    5e-324,
    1e-300,
    1e-10,
    1e-6,
    1e-5,
    nextafter(1e-5, 1),  # series switch
    1e-4,
    nextafter(1e-4, 1),  # trigamma switch
    1e-3,
    0.1,
    0.5,
    1.0,
    2.0,
    4.9,
    5.0,
    8.4,
    8.5,  # asymptotic series switch
    nextafter(8.5, 0),
    nextafter(8.5, 9),
    10.0,
    100.0,
    1e4,
    1e8,
    1e15,
    1e100,
    1e300,
    1e308,
    INF,
    NAN,
    0.0,
    -0.0,
    -1e-300,
    -1e-10,
    -1e-5,
    -0.1,
    -0.5,
    -1.0,  # reflection formula, poles
    -2.0,
    -1.0 + 1e-8,
    -1.0 - 1e-8,
    -1.5,
    -2.0 + 1e-12,
    -3.0 - 1e-6,
    -10.5,
    -100.3,
    -1e5 - 0.5,
    -1e10 - 0.5,
    -1e15,
    -INF,
]

POLYGAMMA_KINDS = [
    # poles raise (Python float division); digamma(tiny) is -inf rather than raising,
    # trigamma raises for x < ~1e-154 where 1 / x**2 divides by an underflowed zero
    ((0.0,), ZeroDivisionError),
    ((-0.0,), ZeroDivisionError),
    ((5e-324,), "nonfinite"),
    ((NAN,), "nan"),
    ((-INF,), "nan"),
    ((INF,), "nonfinite"),
]
TRIGAMMA_KINDS = [
    ((0.0,), ZeroDivisionError),
    ((-0.0,), ZeroDivisionError),
    ((1e-300,), ZeroDivisionError),
    ((-1e-300,), ZeroDivisionError),
    ((5e-324,), ZeroDivisionError),
    ((NAN,), "nan"),
    ((-INF,), "nan"),
    ((INF,), "finite"),
]


class TestPolygamma:
    @pytest.mark.parametrize("name", ["_digamma", "_trigamma"])
    @pytest.mark.parametrize("x", DIGAMMA_POINTS, ids=repr)
    def test_matches_oracle(self, impl, name, x):
        parity(impl, name, (x,))

    @pytest.mark.parametrize(
        ("name", "args", "expected"),
        table("_digamma", POLYGAMMA_KINDS) + table("_trigamma", TRIGAMMA_KINDS),
    )
    def test_poles_and_nonfinite(self, impl, name, args, expected):
        parity(impl, name, args, expected)


BETALN_POINTS = [
    (1e-300, 1e-300),
    (1e-10, 1e-10),
    (1e-10, 1e10),  # lgamma cancellation
    (1e10, 1e-10),
    (1e-5, 1e5),
    (1e8, 1e-8),
    (1.0, 1.0),
    (0.5, 0.5),
    (1e5, 1e5),
    (1e15, 1e15),
    (1e100, 1e100),
    (1e300, 1e300),
    (1e305, 1e305),
    (1e308, 1e308),  # lgamma overflows
    (1e308, 1.0),
    (5e-324, 1.0),
    (1e-320, 1.0),
    (0.0, 1.0),
    (1.0, 0.0),
    (-1.0, 2.0),
    (-0.5, 2.0),
    (NAN, 1.0),
    (1.0, INF),
    (INF, INF),
]


class TestBetaln:
    @pytest.mark.parametrize(("p", "q"), BETALN_POINTS, ids=repr)
    def test_matches_oracle(self, impl, p, q):
        parity(impl, "_betaln", (p, q))


# `_gammainc_der(p, x)`: regimes are x == 0, x > 1e8, p > 1e3 (gaussian),
# x <= 1 or x < p (series) and otherwise a continued fraction.
GAMMAINC_DER_POINTS = [
    (1.0, 0.0),
    (1.0, 5e-324),
    (1.0, 1e-300),
    (1.0, 1.0),
    (1.0, nextafter(1.0, 2)),
    (0.5, 1.0),
    (0.5, nextafter(1.0, 2)),
    (2.0, 1.5),
    (2.0, 2.0),
    (2.0, nextafter(2.0, 3)),
    (2.0, nextafter(2.0, 0)),
    (5.0, 3.0),
    (5.0, 5.0),
    (5.0, nextafter(5.0, 6)),
    (5.0, 50.0),
    (5.0, 1e3),
    (5.0, 1e8),
    (5.0, nextafter(1e8, 2e8)),
    (5.0, 1e8 + 1),
    (5.0, INF),
    (1e-3, 1e-3),
    (1e-3, 1.0),
    (1e-3, 10.0),
    (1e-10, 1e-10),
    (1e-300, 1e-300),
    (100.0, 100.0),
    (100.0, 300.0),
    (999.9, 1000.0),
    (1000.0, 1000.0),
    (nextafter(1000.0, 0), 1000.0),
    (nextafter(1000.0, 2000), 1000.0),
    (1000.5, 900.0),
    (1000.5, 1100.0),
    (2000.0, 2000.0),
    (2000.0, 2300.0),
    (2000.0, 0.0),
    (2000.0, 1.0),
    (1e5, 1e5),
    (1e5, 1.01e5),
    (1e7, 1e7),
    (1e12, 1e12),
    (1e300, 1.0),
    (1e300, 1e300),
    (INF, 1.0),
    (10.0, 1e-30),
    (10.0, 0.5),
    (1.0, 1e-30),
    (50.0, 1e-3),
]

GAMMAINC_DER_DOMAIN = [
    ((-1.0, 1.0), AssertionError),
    ((0.0, 1.0), AssertionError),
    ((-0.0, 1.0), AssertionError),
    ((1.0, -1.0), AssertionError),
    ((1.0, -5e-324), AssertionError),
    ((NAN, 1.0), AssertionError),
    ((1.0, NAN), AssertionError),
]


class TestGammaincDer:
    @pytest.mark.parametrize(("p", "x"), GAMMAINC_DER_POINTS, ids=repr)
    def test_matches_oracle(self, impl, p, x):
        parity(impl, "_gammainc_der", (p, x))

    @pytest.mark.parametrize(
        ("name", "args", "expected"), table("_gammainc_der", GAMMAINC_DER_DOMAIN)
    )
    def test_domain(self, impl, name, args, expected):
        parity(impl, name, args, expected)


def gammainc_inv_grid():
    shapes = [1e-300, 1e-100, 1e-10, 1e-3, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0]
    shapes += [1e3, 1e4, 1e5, 1e6, 1e8, 1e10, 1e15, 1e300, INF, 0.0, -1.0, NAN]
    quantiles = [0.0, 5e-324, 1e-300, 1e-100, 1e-20, 1e-10, 1e-3, 0.5, 0.999]
    quantiles += [1 - 1e-10, nextafter(1.0, 0), 1.0, nextafter(1.0, 2), -0.1, NAN]
    return [(a, q) for a in shapes for q in quantiles]


class TestGammaincInv:
    """
    `_gammainc_inv` is SciPy's ``gammaincinv`` (cephes ``igami``), which the C
    library calls; checked exactly, including away from the interior of the
    domain (the result is a NaN, 0 or inf for invalid shapes and quantiles, with no
    exception).
    """

    @pytest.mark.parametrize(("a", "q"), gammainc_inv_grid(), ids=repr)
    def test_matches_oracle(self, impl, a, q):
        parity(impl, "_gammainc_inv", (a, q))


HYPERU_DOMAIN = [
    ((1.0, 1.0, 1.0), "finite"),  # b == a is allowed
    ((1.0, nextafter(1.0, 0), 1.0), AssertionError),  # b < a
    ((0.0, 1.0, 1.0), AssertionError),
    ((-0.0, 1.0, 1.0), AssertionError),
    ((-1.0, 1.0, 1.0), AssertionError),
    ((5e-324, 1.0, 1.0), "finite"),
    ((1.0, 2.0, 0.0), AssertionError),
    ((1.0, 2.0, -0.0), AssertionError),
    ((1.0, 2.0, -1.0), AssertionError),
    ((1.0, 2.0, 1.0), "finite"),
    ((NAN, 2.0, 1.0), AssertionError),
    ((1.0, NAN, 1.0), AssertionError),
    ((1.0, 2.0, NAN), AssertionError),
    ((INF, INF, 1.0), "nan"),
    ((1.0, INF, 1.0), "nan"),
    # the Laplace root divides by zero when the square root equals t in floating point
    ((1.0, 2.0, INF), ZeroDivisionError),
    ((1.0, 2.0, 1e300), ZeroDivisionError),
    ((1.0, 2.0, 1e-300), ZeroDivisionError),
    ((1.0, 2.0, 5e-324), ZeroDivisionError),
    ((1e300, 1e300, 1.0), ZeroDivisionError),
    ((1.0, 1e10, 1e-3), ZeroDivisionError),
    ((3.6072737850966632e-06, 87694833.655647, 0.10794027431688721), ZeroDivisionError),
    # NOTE: current behaviour, possibly a bug: U(1, 1, x) is finite for
    # small x but the approximation overflows
    ((1.0, 1.0, 1e-300), "nonfinite"),
]

HYP1F1_DOMAIN = [
    ((1.0, 2.0, 0.0), "finite"),
    ((1.0, 2.0, -0.0), "finite"),
    # NOTE: current behaviour, possibly a bug: b == a is allowed by `_valid_hyp1f1`
    # (in approx) but asserted against here
    ((1.0, 1.0, 1.0), AssertionError),
    ((1.0, 0.5, 1.0), AssertionError),
    ((0.0, 1.0, 1.0), AssertionError),
    ((0.0, 1.0, 0.0), AssertionError),  # asserted before the x == 0 shortcut
    ((-1.0, 1.0, 0.0), AssertionError),
    ((NAN, 2.0, 1.0), AssertionError),
    ((1.0, NAN, 1.0), AssertionError),
    ((5e-324, 1.0, 0.0), "finite"),
    ((1.0, nextafter(1.0, 2), 1.0), "nonfinite"),
    ((1.0, 2.0, NAN), "nan"),
    ((1.0, 2.0, INF), "nan"),
    ((1.0, 2.0, -INF), "nan"),
    ((1.0, 2.0, 5e-324), "finite"),
    ((1.0, 2.0, 1e-300), "finite"),
    # NOTE: current behaviour, possibly a bug: -inf for very large x
    ((1.0, 2.0, 1e300), "nonfinite"),
    ((1.0, 2.0, -1e300), "nonfinite"),
    ((1.0, 2.0, 1e15), "nonfinite"),
    ((1e300, 2e300, 1.0), "nonfinite"),
    ((1.0, 1e300, 1.0), "nonfinite"),
    ((1e-300, 1.0, 1.0), "finite"),
    ((1e-10, 2e-10, 1e8), ZeroDivisionError),
]

HYP2F1_DOMAIN = [
    ((1.0, 2.0, 3.0, 0.5), "finite"),
    ((2.0, 1.0, 3.0, 0.5), "finite"),
    ((1.0, 2.0, 2.0, 0.5), "finite"),  # c == b
    ((1.0, 0.5, 1.5, 0.5), "finite"),
    ((0.0, 1.0, 1.0, 0.0), "finite"),  # z == 0 is checked after the assertions
    ((1.0, 2.0, 3.0, -0.0), "finite"),
    ((0.0, 1.0, 1.0, 0.5), AssertionError),  # a = 0: root on boundary
    ((0.0, 1.0, 1.0, -0.5), AssertionError),
    ((0.0, 0.0, 1.0, 0.5), AssertionError),
    ((1.0, 0.0, 1.0, 0.5), AssertionError),  # b = 0, c = a
    ((1.0, 0.0, 2.0, 0.5), "finite"),  # b = 0 is fine when c > a
    ((1.0, 2.0, 1.0, 0.5), AssertionError),  # c == a: root on boundary
    ((1.0, 1.0, 1.0, 0.5), AssertionError),
    ((1.0, 2.0, nextafter(1.0, 0), 0.5), AssertionError),  # c < a
    ((1.0, 2.0, 0.0, 0.5), AssertionError),
    ((1.0, 2.0, -1.0, 0.5), AssertionError),
    ((-1.0, 2.0, 3.0, 0.5), AssertionError),  # a < 0
    ((1.0, -1.0, 3.0, 0.5), AssertionError),  # b < 0
    ((1.0, 2.0, 3.0, 1.0), AssertionError),  # z >= 1
    ((1.0, 2.0, 3.0, nextafter(1.0, 2)), AssertionError),
    ((1.0, 2.0, 3.0, 1.5), AssertionError),
    ((1.0, 2.0, 3.0, nextafter(1.0, 0)), "finite"),
    ((1.0, 2.0, 3.0, NAN), AssertionError),
    ((NAN, 2.0, 3.0, 0.5), AssertionError),
    ((1.0, 2.0, 3.0, -INF), AssertionError),
    ((1.0, 2.0, 3.0, -1e300), AssertionError),  # z / (z - 1) rounds to 1.0
    ((1.0, 2.0, 3.0, -1e16), AssertionError),
    ((1.0, 2.0, 3.0, -9e15), "finite"),
    ((0.0, 1.0, 1e-300, 0.5), ZeroDivisionError),
    ((1e-300, 1.0, 1.0, 0.5), "finite"),
    ((1.0, 1e-300, 1.0, 0.5), AssertionError),
    ((1e300, 1e300, 1e300, 0.5), AssertionError),
    ((1.0, 2.0, 1e300, 0.5), AssertionError),
    ((1.0, 1e300, 2.0, 0.5), AssertionError),
]


def laplace_grid(name):
    if name == "_hyperu_laplace":
        return [
            (a, a + db, x)
            for a in [1e-10, 1e-3, 1.0, 50.0, 1e5]
            for db in [0.0, 1e-10, 1.0, 1e3, 1e10]
            for x in [1e-12, 1e-3, 1.0, 1e3, 1e8, 1e15]
        ]
    if name == "_hyp1f1_laplace":
        return [
            (a, a + db, x)
            for a in [1e-10, 1e-3, 1.0, 50.0, 1e5]
            for db in [1e-10, 1.0, 1e3, 1e10]
            for x in [-1e8, -1e3, -1.0, -1e-10, 1e-10, 1.0, 1e3, 1e8, 1e15]
        ]
    zs = [5e-324, 1e-300, -1e-12, 1e-8, -1e-4, 0.5, 0.9, 1 - 1e-4, 1 - 1e-6]
    zs += [-1.0, -100.0, -1e9]
    abcs = [(1.0, 2.0, 3.0), (5.0, 7.0, 9.0), (0.5, 30.0, 1.5), (30.0, 40.0, 35.0)]
    return [(*abc, z) for abc in abcs for z in zs]


LAPLACE = {
    "_hyperu_laplace": HYPERU_DOMAIN,
    "_hyp1f1_laplace": HYP1F1_DOMAIN,
    "_hyp2f1_laplace": HYP2F1_DOMAIN,
}


def unity_formula(a, b, c, z):
    """
    The result of `_hyp2f1_laplace` in the unity limit, written out. See DLMF 15.4:
    the sign of ``g = c - a - b`` selects the limit, and negative ``z`` is mapped to
    ``z / (z - 1)`` close to one by Pfaff's transformation.
    """
    s = 0.0
    if z < 0.0:
        s = -b * log(1 - z)
        a = c - a
        z = z / (z - 1)
    g = c - a - b
    if g < 0.0:
        value = lgamma(c) + lgamma(-g) - lgamma(a) - lgamma(b) + g * log(1 - z)
    elif g > 0.0:
        value = lgamma(c) + lgamma(g) - lgamma(c - a) - lgamma(c - b)
    else:
        value = log(-log(1 - z)) + lgamma(a + b) - lgamma(a) - lgamma(b)
    return s + value


def unity_threshold():
    """
    Largest ``x < 1`` for which ``np.isclose(x, 1.0)`` fails, and the next one up
    """
    lo, hi = 0.5, 1.0 - 1e-9  # not close, close
    while nextafter(lo, 1.0) != hi:
        mid = lo + (hi - lo) / 2
        if np.isclose(mid, 1.0):
            hi = mid
        else:
            lo = mid
    return lo, hi


def negative_unity_threshold():
    """
    As `unity_threshold`, for ``z < 0`` mapped to ``z / (z - 1)``
    """
    lo, hi = -1e3, -1e9  # not close, close
    while nextafter(lo, hi) != hi:
        mid = lo + (hi - lo) / 2
        if np.isclose(mid / (mid - 1), 1.0):
            hi = mid
        else:
            lo = mid
    return lo, hi


def unity_args_from_ep(a_i, b_i, a_j, b_j, y, mu):
    """
    Arguments of `_hyp2f1_laplace` in the edge update of EP, for hugely negative z
    """
    return a_j, a_i + a_j + y, a_j + y + 1, (mu - b_j) / (mu + b_i)


# (a, b, c, z) for the three limits (g < 0, g > 0, g == 0), for z close to one and,
# via Pfaff's transformation, for large negative z (where g = a - b)
UNITY_CASES = [
    (2.0, 3.0, 4.0, 1 - 1e-8),
    (1.0, 2.0, 5.0, 1 - 1e-8),
    (1.0, 2.0, 3.0, 1 - 1e-8),
    (2.0, 3.0, 4.0, -1e8),
    (3.0, 2.0, 5.0, -1e8),
    (2.0, 2.0, 5.0, -1e8),
    (19.0, 30.0, 25.0, 1 - 1e-7),
    (1.0, 2.0, 1.0 + 2.0 + 1e-3, 1 - 1e-9),  # g tiny positive
    (1.0, 2.0, 3.0 - 1e-3, 1 - 1e-9),  # g tiny negative
    # parameters of EP updates that reach the limit
    unity_args_from_ep(202.865, 0.0153, 0.0012724, 101353547.00, 0.0, 2.404560e-05),
    unity_args_from_ep(999.998, 0.1041, 0.0012612, 14749022.000, 2.0, 7.830300e-05),
    unity_args_from_ep(157.778, 0.0182, 0.0011993, 20861958.000, 0.0, 4.128000e-07),
    unity_args_from_ep(1000.00, 0.1431, 0.0009718, 22638776.000, 0.0, 5.881110e-05),
    unity_args_from_ep(405.482, 0.1484, 0.0009711, 41053809.000, 0.0, 6.229410e-05),
    unity_args_from_ep(115.773, 0.7506, 0.0015549, 23315696857.0, 0.0, 3.601680e-05),
    unity_args_from_ep(73.4909, 1.0600, 0.0010028, 56254280956.0, 0.0, 0.0002302392),
    unity_args_from_ep(18.9981, 0.1315, 0.0015053, 23443723634.0, 1.0, 2.470348e-05),
]


class TestLaplaceDomain:
    """
    Edge cases of `_hyperu_laplace`, `_hyp1f1_laplace` and `_hyp2f1_laplace`: the
    domain assertions, the zero shortcuts (``x == 0`` returns 0 after the checks), the
    division by zero when a Laplace root cancels, and overflow.
    """

    @pytest.mark.parametrize(
        ("name", "args", "expected"),
        [p for name, rows in LAPLACE.items() for p in table(name, rows)],
    )
    def test_domain(self, impl, name, args, expected):
        parity(impl, name, args, expected)

    @pytest.mark.parametrize(
        ("name", "args"),
        [
            pytest.param(name, args, id=f"{name}{args}")
            for name in LAPLACE
            for args in laplace_grid(name)
        ],
    )
    def test_grid(self, impl, name, args):
        parity(impl, name, args)


class TestHyp2f1Unity:
    """
    The unity limit (`_hyp2f1_unity`, reached when ``z`` is ``np.isclose`` to one)
    """

    @pytest.mark.parametrize("args", UNITY_CASES, ids=repr)
    def test_unity_branches(self, impl, args):
        a, b, c, z = args
        mapped = z / (z - 1) if z < 0 else z
        assert np.isclose(mapped, 1.0), "case must take the unity branch"
        result = parity(impl, "_hyp2f1_laplace", args)
        assert_same_values(result.value, unity_formula(a, b, c, z))

    @pytest.mark.parametrize("sign", ["positive", "negative"])
    @pytest.mark.parametrize("side", ["not_close", "close"])
    def test_switch_point(self, impl, sign, side):
        # the unity formula is used from the first np.isclose argument onwards
        if sign == "positive":
            not_close, close = unity_threshold()
        else:
            not_close, close = negative_unity_threshold()
        z = close if side == "close" else not_close
        parity(impl, "_hyp2f1_laplace", (5.0, 7.0, 9.0, z), "finite")


def draw_polygamma(rng, extreme):
    if extreme:
        return (float(rng.choice([-1.0, 1.0]) * log_uniform(rng, -12, 12)),)
    x = log_uniform(rng, -7, 3)
    if rng.random() < 0.15:
        x = -rng.uniform(0.01, 20)  # reflection formula
    return (float(x),)


def draw_betaln(rng, extreme):
    lo, hi = (-8, 8) if extreme else (-2, 3)
    return log_uniform(rng, lo, hi), log_uniform(rng, lo, hi)


def draw_gammainc_der(rng, extreme):
    if extreme:
        return log_uniform(rng, -8, 8), log_uniform(rng, -8, 8)
    # series, continued fraction and gaussian regimes
    return log_uniform(rng, -1, 3.5), log_uniform(rng, -3, 4)


def draw_gammainc_inv(rng, extreme):
    if extreme:
        return log_uniform(rng, -8, 8), float(rng.uniform(0, 1))
    return log_uniform(rng, -1, 3), float(rng.uniform(0.001, 0.999))


def draw_hyperu(rng, extreme):
    # requires b >= a > 0 and x > 0
    if extreme:
        a = log_uniform(rng, -8, 8)
        return a, a + log_uniform(rng, -8, 8), log_uniform(rng, -12, 12)
    a = log_uniform(rng, -1, 2)
    return a, a + log_uniform(rng, -2, 2), log_uniform(rng, -3, 3)


def draw_hyp1f1(rng, extreme):
    # requires b > a > 0, x of either sign
    sign = float(rng.choice([-1.0, 1.0]))
    if extreme:
        a = log_uniform(rng, -8, 8)
        return a, a + log_uniform(rng, -8, 8), sign * log_uniform(rng, -12, 12)
    a = log_uniform(rng, -1, 2)
    return a, a + log_uniform(rng, -2, 2), sign * log_uniform(rng, -2, 2)


def draw_hyp2f1(rng, extreme):
    if extreme:
        a = log_uniform(rng, -8, 8)
        b = log_uniform(rng, -8, 8)
        c = a + (0.0 if rng.random() < 0.3 else log_uniform(rng, -8, 8))
        z = float(rng.choice([-1.0, 1.0, 1.0]) * log_uniform(rng, -14, 0))
        return a, b, c, z + 1.0 if rng.random() < 0.3 else z
    # parameters of the edge (c = a_j + y + 1) and unphased (c = a_i + a_j) updates,
    # with z < 1 on either side of zero and close to unity
    a_i, a_j = rng.uniform(0.5, 30, 2)
    y = float(rng.integers(0, 12))
    c = a_j + y + 1 if rng.random() < 0.5 else a_i + a_j
    u = rng.random()
    if u < 0.4:
        z = -log_uniform(rng, -3, 3)
    elif u < 0.8:
        z = 1 - log_uniform(rng, -6, 0)
    else:
        z = 1 - log_uniform(rng, -12, -9)
    return float(a_j), float(a_i + a_j + y), float(c), z


DRAWS = {
    "_digamma": draw_polygamma,
    "_trigamma": draw_polygamma,
    "_betaln": draw_betaln,
    "_gammainc_der": draw_gammainc_der,
    "_gammainc_inv": draw_gammainc_inv,
    "_hyperu_laplace": draw_hyperu,
    "_hyp1f1_laplace": draw_hyp1f1,
    "_hyp2f1_laplace": draw_hyp2f1,
}


class TestRandomDraws:
    """
    Seeded draws of arguments, in ranges typical of EP and over 16 orders of magnitude
    ("extreme", where exceptions and non-finite values are common). The failing
    draw is named in the assertion.
    """

    @pytest.mark.parametrize("seed", [1, 2, 3])
    @pytest.mark.parametrize("mode", ["typical", "extreme"])
    @pytest.mark.parametrize("name", list(DRAWS))
    def test_random_draws(self, impl, name, mode, seed):
        rng = np.random.default_rng(seed)
        for i in range(40):
            args = DRAWS[name](rng, mode == "extreme")
            try:
                parity(impl, name, args)
            except AssertionError as e:
                raise AssertionError(f"draw {i}: {name}{args}: {e}") from None
