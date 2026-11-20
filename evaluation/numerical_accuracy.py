"""
Numerical accuracy of the special functions and gamma approximations in tsdate
(``tsdate.hypergeo`` and ``tsdate.approx``), per function and regime. Each row of the
table reports the worst error of the production function against a reference (mpmath,
SciPy, a closed form) or the violation of a property the approximation should have
(symmetry, monotonicity, continuity across a switch point). These are measurements, not
tests: the unit tests only check that implementations agree with the numba oracles, so
inaccuracy of the algorithms themselves is recorded here. Behaviour that looks wrong
is a candidate for a fix.

Errors are relative unless the metric says "abs" (log-scale quantities such as
log U, log 1F1 and log 2F1 are compared in absolute terms), "ulp" or "count".

Example:
  python evaluation/numerical_accuracy.py --quick
  python evaluation/numerical_accuracy.py --only hyp2f1 gammainc --output acc.csv
"""

import argparse
import csv
import math
import pathlib

import mpmath
import numpy as np
import scipy.special

from tsdate import approx, hypergeo

INF = math.inf
RESULTS = pathlib.Path(__file__).parent / "results"


def call(func, *args):
    """Call a function, or the ``ctypes`` interface of a compiled numba ``cfunc``"""
    return getattr(func, "ctypes", func)(*args)


def try_call(func, *args):
    """The result of ``func``, or None if it raised"""
    try:
        return call(func, *args)
    except Exception:  # noqa: BLE001
        return None


def log_uniform(rng, lo, hi, size=None):
    return 10.0 ** rng.uniform(lo, hi, size)


class Report:
    """Collects rows of (group, function, regime, metric, value, points, note)"""

    def __init__(self):
        self.rows = []

    def add(self, group, function, regime, metric, value, n=1, note=""):
        self.rows.append((group, function, regime, metric, value, n, note))

    def worst(self, group, function, regime, metric, errors, note=""):
        """Worst of ``errors`` (None marks a call that raised)"""
        raised = sum(e is None for e in errors)
        finite = [e for e in errors if e is not None and math.isfinite(e)]
        value = max(finite) if finite else math.nan
        nonfinite = len(errors) - raised - len(finite)
        if raised or nonfinite:
            note = f"{note} raised={raised} nonfinite={nonfinite}".strip()
        self.add(group, function, regime, metric, value, len(errors), note)

    def print(self):
        header = ("function", "regime", "metric", "value", "n", "note")
        body = [(r[1], r[2], r[3], f"{r[4]:.3g}", str(r[5]), r[6]) for r in self.rows]
        widths = [max(len(str(x)) for x in col) for col in zip(header, *body)]
        group = None
        for row, raw in zip(body, self.rows):
            if raw[0] != group:
                group = raw[0]
                print(f"\n== {group} ==")
                print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
            print("  ".join(x.ljust(w) for x, w in zip(row, widths)).rstrip())

    def write(self, path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(("group", *("function regime metric value n note".split())))
            writer.writerows(self.rows)


def rel(actual, expected):
    if actual is None:
        return None
    if actual == expected:
        return 0.0
    return abs(actual - expected) / abs(expected)


def ulp(actual, expected):
    if actual is None:
        return None
    return abs(actual - expected) / np.spacing(abs(expected))


# --- polygamma, betaln --- #


def mp_psi(n, x):
    with mpmath.workdps(60):
        return float(mpmath.psi(n, x))


POLYGAMMA_REGIMES = {
    "x <= 1e-5": [1e-10, 1e-6, 1e-5],
    "1e-5 < x < 8.5 (recurrence)": [1e-3, 0.1, 1.0, 2.0, 4.9, 5.0, 8.4],
    "x >= 8.5 (asymptotic)": [8.5, 100.0, 1e4, 1e8],
    "x < 0 (reflection)": [-1e-5, -0.5, -1.5, -10.5, -100.3, -1e5 - 0.5],
    "near a negative integer": [-1.0 + 1e-8, -1.0 - 1e-8, -3.0 - 1e-6, -2.0 + 1e-12],
}


def polygamma(report):
    for name, order in [("_digamma", 0), ("_trigamma", 1)]:
        func = getattr(hypergeo, name)
        for regime, xs in POLYGAMMA_REGIMES.items():
            errors = [rel(try_call(func, x), mp_psi(order, x)) for x in xs]
            report.worst("polygamma", name, regime, "rel", errors)
    # trigamma returns 1 / x**2 for x <= 1e-4, omitting pi**2 / 6
    xs = [1e-6, 1e-5, 1e-4]
    errors = [rel(try_call(hypergeo._trigamma, x), mp_psi(1, x)) for x in xs]
    report.worst("polygamma", "_trigamma", "x <= 1e-4 (1/x**2)", "rel", errors)
    # continuity at the switch points of the series
    for x in [1e-5, 8.5]:
        lo = hypergeo._digamma(x)
        hi = hypergeo._digamma(np.nextafter(x, 2 * x))
        report.add("polygamma", "_digamma", f"jump at x={x}", "rel", rel(lo, hi))


def betaln(report):
    regimes = {
        "balanced": [(0.5, 0.5), (1.0, 1.0), (1e5, 1e5), (1e15, 1e15), (1e300, 1e300)],
        "tiny and tiny": [(1e-10, 1e-10), (1e-300, 1e-300)],
        "tiny and huge (cancellation)": [(1e-10, 1e10), (1e8, 1e-8), (1e-5, 1e5)],
    }
    for regime, points in regimes.items():
        errors = []
        for p, q in points:
            errors.append(
                rel(try_call(hypergeo._betaln, p, q), scipy.special.betaln(p, q))
            )
        report.worst("betaln", "_betaln", regime, "rel", errors)


# --- incomplete gamma --- #


def gammainc_der_reference(p, x):
    with mpmath.workdps(40):
        return float(mpmath.diff(lambda s: mpmath.gammainc(s, 0, x, regularized=True), p))


def gammainc_der(report):
    regimes = {
        "series (x <= 1 or x < p)": [(1.0, 1.0), (0.5, 1.0), (1e-3, 1.0), (10.0, 0.5)]
        + [(2.0, 1.5), (5.0, 3.0), (100.0, 100.0 - 1e-9)],
        "continued fraction (x >= p, x > 1)": [(5.0, 5.0), (5.0, 50.0), (100.0, 100.0)]
        + [(1e-3, 10.0)],
        "just above x = 1 (continued fraction)": [(1.0, np.nextafter(1.0, 2))],
        "gaussian (p > 1000)": [(1000.5, 900.0), (2000.0, 2000.0), (2000.0, 2300.0)],
    }
    for regime, points in regimes.items():
        errors = []
        for p, x in points:
            ref = gammainc_der_reference(p, x)
            errors.append(rel(try_call(hypergeo._gammainc_der, p, x), ref))
        report.worst("gammainc", "_gammainc_der", regime, "rel", errors)
    below = hypergeo._gammainc_der(1000.0, 1000.0)
    above = hypergeo._gammainc_der(np.nextafter(1000.0, 2000.0), 1000.0)
    report.add("gammainc", "_gammainc_der", "jump at p = 1000", "rel", rel(above, below))


def exact_gammainc_inv(a, q, x0):
    """Root of the regularised lower incomplete gamma minus q, near ``x0``; or None"""
    with mpmath.workdps(40):

        def f(t):
            return mpmath.gammainc(a, 0, t, regularized=True) - q

        try:
            bracket = (x0 * (1 - 1e-6), x0 * (1 + 1e-6))
            return float(mpmath.re(mpmath.findroot(f, bracket, solver="illinois")))
        except (ValueError, ZeroDivisionError):
            return None


def gammainc_inv(report, rng, quick):
    func = hypergeo._gammainc_inv
    n = 10 if quick else 40
    for a in [0.1, 0.3, 1.0, 2.5, 10.0, 60.0, 1e3, 1e4]:
        errors = []
        for q in log_uniform(rng, -12, -0.01, n):
            x = try_call(func, a, q)
            errors.append(rel(scipy.special.gammainc(a, x), q) if x else None)
        report.worst("gammainc", "_gammainc_inv", f"roundtrip, a={a}", "rel", errors)
    # relative to an exact inverse
    bands = {"a < 0.1": (-2, -1), "0.1 <= a < 1": (-1, 0), "1 <= a < 10": (0, 1)}
    bands |= {"10 <= a < 100": (1, 2), "a >= 100": (2, 4)}
    n = 8 if quick else 60
    for band, (lo, hi) in bands.items():
        errors = []
        for a, q in zip(log_uniform(rng, lo, hi, n), rng.uniform(1e-8, 1.0, n)):
            x = try_call(func, a, q)
            exact = exact_gammainc_inv(a, q, x)
            if exact is None:
                continue
            errors.append(ulp(x, exact))
        errors = [e for e in errors if e is not None]
        if errors:
            report.add(
                "gammainc",
                "_gammainc_inv",
                band,
                "ulp median",
                np.median(errors),
                len(errors),
            )
            report.add(
                "gammainc", "_gammainc_inv", band, "ulp max", max(errors), len(errors)
            )
    qs = np.linspace(0.001, 0.999, 50)
    violations = 0
    for a in [0.2, 1.0, 5.0, 100.0]:
        x = [call(func, a, q) for q in qs.tolist()]
        violations += int(np.sum(np.diff(x) <= 0))
    report.add("gammainc", "_gammainc_inv", "monotone in q", "count", violations, 200)
    exact = [(q, call(func, 1.0, q), -math.log1p(-q)) for q in [1e-100, 1e-3, 0.5, 0.999]]
    errors = [rel(x, e) for _, x, e in exact]
    report.worst("gammainc", "_gammainc_inv", "a = 1, exponential", "rel", errors)


# --- Laplace approximations to hypergeometric functions --- #


def log_hyperu(a, b, x):
    with mpmath.workdps(30):
        return float(mpmath.log(mpmath.hyperu(a, b, x, maxterms=10**6)))


def log_hyp1f1(a, b, x):
    with mpmath.workdps(30):
        return float(mpmath.log(mpmath.hyp1f1(a, b, x, maxterms=10**6)))


def log_hyp2f1(a, b, c, z):
    with mpmath.workdps(30):
        return float(mpmath.log(mpmath.re(mpmath.hyp2f1(a, b, c, z, maxterms=10**7))))


def absolute(func, reference, points):
    errors = []
    for args in points:
        value = try_call(func, *args)
        if isinstance(value, tuple):
            value = value[0]
        errors.append(None if value is None else abs(value - reference(*args)))
    return errors


def hyperu(report):
    func = hypergeo._hyperu_laplace
    regimes = {
        "a >= 1 (EP: a = y + 1)": [(1.0, 2.0, 1.0), (2.0, 5.0, 0.5), (5.0, 15.0, 1.0)]
        + [(10.0, 20.0, 50.0), (50.0, 60.0, 1.0), (1000.0, 1010.0, 1e4)],
        "large b - a": [(1.0, 11.0, 1e-2), (1.0, 101.0, 100.0), (50.0, 1050.0, 100.0)],
        "b == a": [(1.0, 1.0, 1.0)],
        "a < 1 (Stirling)": [(1e-3, 10.001, 1.0), (0.1, 10.1, 1.0), (1.0, 1.001, 1e-6)],
    }
    for regime, points in regimes.items():
        report.worst(
            "hyperu", "_hyperu_laplace", regime, "abs", absolute(func, log_hyperu, points)
        )
    errors = []
    for a, b, x in [
        (1.0, 2.0, 1.0),
        (2.0, 5.0, 0.5),
        (1.0, 11.0, 1e-2),
        (10.0, 20.0, 50.0),
    ]:
        step = 1e-6 * x
        fd = (func(a, b, x + step)[0] - func(a, b, x - step)[0]) / (2 * step)
        errors.append(rel(func(a, b, x)[1], fd))
    report.worst(
        "hyperu", "_hyperu_laplace", "derivative vs finite difference", "rel", errors
    )
    grid = [
        (a, a + d, x)
        for a in [1.0, 50.0, 1e5]
        for d in [1e-10, 1.0, 1e3]
        for x in [1e-3, 1.0, 1e3]
    ]
    values = [try_call(func, *args) for args in grid]
    positive = sum(v is not None and v[1] > 0 for v in values)
    report.add(
        "hyperu",
        "_hyperu_laplace",
        "d/dx log U > 0, a >= 1",
        "count",
        positive,
        len(grid),
    )
    d = func(1e-10, 1.0 + 1e-10, 1e-12)[1]
    report.add(
        "hyperu",
        "_hyperu_laplace",
        "d/dx log U at a = 1e-10",
        "value",
        d,
        1,
        "should be < 0",
    )
    values = [func(2.0, 2.0, x)[0] for x in [0.5, 1.0, 2.0, 4.0]]
    report.add(
        "hyperu",
        "_hyperu_laplace",
        "U(a, a, x) decreasing",
        "count",
        int(np.sum(np.diff(values) >= 0)),
        3,
    )


def hyp1f1(report):
    func = hypergeo._hyp1f1_laplace
    regimes = {
        "moderate": [
            (1.0, 2.0, 1.0),
            (1.0, 2.0, -1.0),
            (5.0, 8.0, 10.0),
            (5.0, 8.0, -10.0),
        ],
        "large shapes": [(50.0, 1000.0, 100.0), (50.0, 1000.0, -100.0)]
        + [(1000.0, 1010.0, 20.0), (1000.0, 1010.0, -20.0)],
        "a -> 0": [(1e-10, 1.0 + 1e-10, 1e3), (1e-10, 2e-10, 1.0)],
    }
    for regime, points in regimes.items():
        report.worst(
            "hyp1f1", "_hyp1f1_laplace", regime, "abs", absolute(func, log_hyp1f1, points)
        )
    errors = []
    for a, b, x in [(5.0, 8.0, 10.0), (1.0, 3.0, 2.0), (30.0, 40.0, 100.0)]:
        errors.append(rel(func(a, b, -x), -x + func(b - a, b, x)))
    report.worst("hyp1f1", "_hyp1f1_laplace", "Kummer transformation", "rel", errors)
    xs = np.linspace(-50, 50, 41).tolist()
    values = [func(2.0, 5.0, x) for x in xs]
    report.add(
        "hyp1f1",
        "_hyp1f1_laplace",
        "increasing in x",
        "count",
        int(np.sum(np.diff(values) <= 0)),
        40,
    )


def hyp2f1(report, quick):
    func = hypergeo._hyp2f1_laplace
    zs = [(-1e9,), (-100.0,), (-1.0,), (0.5,), (0.9,), (0.99,), (0.99998,), (0.999999,)]
    abcs = {
        "(1, 2, 3)": (1.0, 2.0, 3.0),
        "(5, 7, 9)": (5.0, 7.0, 9.0),
        "(30, 40, 35)": (30.0, 40.0, 35.0),
        "(100, 200, 150)": (100.0, 200.0, 150.0),
        "(0.5, 30, 1.5)": (0.5, 30.0, 1.5),
        "(1, 2, 5) g > 0": (1.0, 2.0, 5.0),
        "(2, 3, 5) g = 0": (2.0, 3.0, 5.0),
        "(3, 2, 5)": (3.0, 2.0, 5.0),
        "(2, 2, 5)": (2.0, 2.0, 5.0),
        "(0.1, 0.2, 1) small": (0.1, 0.2, 1.0),
    }
    for label, abc in abcs.items():
        for (z,) in zs:
            if abc == (100.0, 200.0, 150.0) and z > 0.9999:
                continue
            if quick and z not in (-100.0, 0.5, 0.99998):
                continue
            errors = absolute(func, log_hyp2f1, [(*abc, z)])
            regime = f"(a,b,c)={label}, z={z:g}"
            report.worst("hyp2f1", "_hyp2f1_laplace", regime, "abs", errors)
    # tiny z: 2F1 ~ 1 + a b z / c
    errors = []
    for a, b, c in [
        (1.0, 2.0, 3.0),
        (5.0, 7.0, 9.0),
        (0.5, 30.0, 1.5),
        (30.0, 40.0, 35.0),
    ]:
        for z in [1e-12, 1e-8, 1e-4, -1e-12, -1e-8, -1e-4]:
            errors.append(abs(func(a, b, c, z) - a * b / c * z))
    report.worst("hyp2f1", "_hyp2f1_laplace", "|z| <= 1e-4 vs a b z / c", "abs", errors)
    # the unity limit, through hugely negative z (the parameters of EP updates)
    pars = [
        (202.865, 0.0153, 0.0012724, 101353547.00, 0.0, 2.404560e-05),
        (999.998, 0.1041, 0.0012612, 14749022.000, 2.0, 7.830300e-05),
        (73.4909, 1.0600, 0.0010028, 56254280956.0, 0.0, 0.0002302392),
        (18.9981, 0.1315, 0.0015053, 23443723634.0, 1.0, 2.470348e-05),
    ]
    errors = []
    for a_i, b_i, a_j, b_j, y, mu in pars:
        z = (mu - b_j) / (mu + b_i)
        args = (a_j, a_i + a_j + y, a_j + y + 1, z)
        errors.append(abs(func(*args) - log_hyp2f1(*args)))
    report.worst("hyp2f1", "_hyp2f1_laplace", "unity limit, EP parameters", "abs", errors)
    # discontinuity where np.isclose(z, 1) switches to the unity limit
    lo, hi = 0.5, 1.0 - 1e-9
    while np.nextafter(lo, 1.0) != hi:
        mid = lo + (hi - lo) / 2
        lo, hi = (lo, mid) if np.isclose(mid, 1.0) else (mid, hi)
    for abc in [(1.0, 2.0, 3.0), (30.0, 40.0, 35.0)]:
        jump = func(*abc, hi) - func(*abc, lo)
        report.add(
            "hyp2f1",
            "_hyp2f1_laplace",
            f"jump at switch to unity, {abc}",
            "abs",
            abs(jump),
        )
    sym = [rel(func(2.0, 5.0, 7.0, z), func(5.0, 2.0, 7.0, z)) for z in [-3.0, 0.2, 0.9]]
    report.worst("hyp2f1", "_hyp2f1_laplace", "symmetric in a, b", "rel", sym)
    values = [func(3.0, 4.0, 8.0, z) for z in np.linspace(-20, 0.98, 60)]
    report.add(
        "hyp2f1",
        "_hyp2f1_laplace",
        "increasing in z",
        "count",
        int(np.sum(np.diff(values) <= 0)),
        59,
    )


# --- gamma approximations (tsdate.approx) --- #


def gamma_quantiles(shape, rate, q1=0.25, q2=0.75):
    x1, x2 = scipy.special.gammaincinv(shape, [q1, q2]) / rate
    return q1, q2, float(x1), float(x2)


def gamma_fits(report, rng, quick):
    kl = approx.approximate_gamma_kl
    errors, residuals = [], []
    for x, gap in [
        (1.0, 1e-3),
        (1.0, 0.5),
        (1.0, 5.0),
        (1e-3, 2.0),
        (1e6, 0.1),
        (3.0, 1e-4),
    ]:
        shape, rate = kl(x, math.log(x) - gap)
        alpha = shape + 1.0
        errors.append(rel(alpha / rate, x))
        residuals.append(abs(scipy.special.digamma(alpha) - math.log(alpha) + gap))
    report.worst("approx", "approximate_gamma_kl", "mean matched", "rel", errors)
    report.worst(
        "approx", "approximate_gamma_kl", "stationarity residual", "abs", residuals
    )
    below = kl(1.0, -np.nextafter(5e-5, 0.0))[0]
    above = kl(1.0, -np.nextafter(5e-5, 1.0))[0]
    report.add(
        "approx",
        "approximate_gamma_kl",
        "jump at asymptotic switch",
        "rel",
        rel(below, above),
    )
    errors = []
    for mean, variance in [(1.0, 1.0), (10.0, 4.0), (1e3, 1e-6), (1e-3, 1e-3)]:
        shape, rate = approx.approximate_gamma_mom(mean, variance)
        errors += [rel((shape + 1) / rate, mean), rel((shape + 1) / rate**2, variance)]
    report.worst(
        "approx", "approximate_gamma_mom", "mean and variance matched", "rel", errors
    )
    mean, variance = 3.0, 0.5
    logx, xlogx, logx2 = approx.approximate_log_moments(mean, variance)
    expected = (
        math.log(mean) - 0.5 * variance / mean**2,
        mean * math.log(mean) + 0.5 * variance / mean,
        math.log(mean) ** 2 + (1 - math.log(mean)) * variance / mean**2,
    )
    errors = [rel(a, e) for a, e in zip((logx, xlogx, logx2), expected)]
    report.worst(
        "approx", "approximate_log_moments", "second order expansion", "rel", errors
    )
    shapes = [0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 49.0, 50.0, 51.0, 999.0, 1001.0, 1e4, 1e6]
    for max_shape in [50.0, 1000.0, 1e6]:
        errors = []
        for shape in shapes:
            args = (*gamma_quantiles(shape, 0.001), max_shape)
            result = try_call(approx.approximate_gamma_iqr, *args)
            errors.append(
                None if result is None else rel(result[0] + 1.0, min(shape, max_shape))
            )
        report.worst(
            "approx",
            "approximate_gamma_iqr",
            f"shape recovered, max_shape={max_shape:g}",
            "rel",
            errors,
        )
    errors = [
        rel(approx.average_gammas(np.full(3, s), np.full(3, r))[0] + 1.0, s + 1.0)
        for s, r in [(3.0, 0.5), (0.5, 2.0)]
    ]
    report.worst("approx", "average_gammas", "identical gammas recovered", "rel", errors)
    shape, rate = approx.average_gammas(np.array([0.0]), np.array([1.0]))
    report.add(
        "approx", "average_gammas", "single gamma, shape 1", "rel", rel(shape + 1.0, 1.0)
    )


def moment_functions(report, quick):
    a_i, b_i, y, mu = 10.0, 0.01, 3.0, 1e-4
    s, r = a_i + y, b_i + 2 * mu
    got = approx.twin_moments(a_i, b_i, y, mu)
    expected = (math.log(2) * y + math.lgamma(s) - math.log(r) * s, s / r, s / r**2)
    report.worst(
        "approx",
        "twin_moments",
        "closed form",
        "rel",
        [rel(g, e) for g, e in zip(got, expected)],
    )
    t_i, t_j = 7.0, 2.0
    mn, va = approx.mutation_edge_moments(t_i, t_j)
    errors = [rel(mn, 4.5), rel(va, 25.0 / 12.0)]
    report.worst("approx", "mutation_edge_moments", "closed form", "rel", errors)
    pr, mn, va = approx.mutation_block_moments(t_i, t_j)
    errors = [rel(pr, 7.0 / 9.0), rel(mn, pr * t_i / 2 + (1 - pr) * t_j / 2)]
    report.worst("approx", "mutation_block_moments", "closed form", "rel", errors)
    # rootward with t_j -> 0 approaches the gamma with shape a_i + y, rate b_i + mu
    a_i, b_i, y, mu = 10.0, 0.01, 1.0, 1e-4
    s, r = a_i + y, b_i + mu
    exact = (math.lgamma(s) - s * math.log(r), s / r, s / r**2)
    for t_j in [1e-10, 1e-3]:
        got = approx.rootward_moments(t_j, a_i, b_i, y, mu)
        report.add(
            "approx",
            "rootward_moments",
            f"t_j={t_j:g} vs t_j=0 closed form, log-lik",
            "abs",
            abs(got[0] - exact[0]),
        )
        report.add(
            "approx",
            "rootward_moments",
            f"t_j={t_j:g} vs t_j=0 closed form, mean",
            "rel",
            rel(got[1], exact[1]),
        )
    # y = 0: U(a, a + 1, z) = z**-a
    a_j, b_j, mu, t_i = 10.0, 0.01, 1e-4, 1000.0
    logl, mn, va = approx.sideways_moments(t_i, a_j, b_j, 0.0, mu)
    z = t_i * (mu + b_j)
    f0 = logl + mu * t_i - a_j * math.log(t_i) - math.lgamma(a_j)
    report.add(
        "approx", "sideways_moments", "y = 0 vs z**-a", "rel", rel(f0, -a_j * math.log(z))
    )
    # projected gammas reproduce the moments
    args = (10.0, 0.01, 10.0, 0.01, 1.0, 1e-4)
    _, mn_i, va_i, mn_j, va_j = approx.moments(*args)
    _, proj_i, proj_j = approx.gamma_projection(
        np.array([9.0, 0.01]), np.array([9.0, 0.01]), np.array([1.0, 1e-4])
    )
    errors = []
    for proj, mn, va in [(proj_i, mn_i, va_i), (proj_j, mn_j, va_j)]:
        errors += [
            rel((proj[0] + 1) / proj[1], mn),
            rel((proj[0] + 1) / proj[1] ** 2, va),
        ]
    report.worst(
        "approx", "gamma_projection", "projection matches moments", "rel", errors
    )
    # variance by cancellation (second moment minus squared mean)
    for name, args in [
        ("moments", (1e8, 1e5, 1e8, 1e5, 1, 1e-4)),
        ("moments", (10, 0.01, 10, 0.01, 1, 1e-4)),
    ]:
        va = getattr(approx, name)(*[float(a) for a in args])[2]
        regime = "shape 1e8" if args[0] == 1e8 else "typical"
        report.add(
            "approx",
            name,
            f"variance of parent age, {regime}",
            "value",
            va,
            1,
            "negative = lost",
        )
    rng = np.random.default_rng(1)
    count, total = 0, 200 if quick else 2000
    for _ in range(total):
        a_i, a_j = rng.uniform(0.5, 30, 2)
        b_i, b_j = log_uniform(rng, -5, -2, 2)
        y, mu = float(rng.integers(0, 12)), log_uniform(rng, -4, -2)
        result = approx.moments(a_i, b_i, a_j, b_j, y, mu)
        count += not (result[2] > 0 and result[4] > 0)
    report.add(
        "approx",
        "moments",
        "non-positive variance, typical EP range",
        "count",
        count,
        total,
    )
    count = 0
    for _ in range(total):
        a_i, a_j = rng.uniform(0.5, 30, 2)
        b_i, b_j = log_uniform(rng, -5, -2, 2)
        y, mu = float(rng.integers(0, 12)), log_uniform(rng, -4, -2)
        pr = approx.mutation_unphased_moments(a_i, b_i, a_j, b_j, y, mu)[0]
        count += not (0 <= pr <= 1)
    report.add(
        "approx",
        "mutation_unphased_moments",
        "probability outside [0, 1], typical EP range",
        "count",
        count,
        total,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--quick", action="store_true", help="fewer points, skip slow rows")
    p.add_argument("--only", nargs="+", default=None, help="substrings of group names")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output", default=None, help="CSV file name, written to results/")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    report = Report()
    steps = {
        "polygamma": lambda: polygamma(report),
        "betaln": lambda: betaln(report),
        "gammainc": lambda: (
            gammainc_der(report),
            gammainc_inv(report, rng, args.quick),
        ),
        "hyperu": lambda: hyperu(report),
        "hyp1f1": lambda: hyp1f1(report),
        "hyp2f1": lambda: hyp2f1(report, args.quick),
        "approx": lambda: (
            gamma_fits(report, rng, args.quick),
            moment_functions(report, args.quick),
        ),
    }
    for name, step in steps.items():
        if args.only is None or any(o in name for o in args.only):
            step()
    report.print()
    if args.output:
        RESULTS.mkdir(exist_ok=True)
        path = RESULTS / args.output
        report.write(path)
        print("\nwrote", path)


if __name__ == "__main__":
    main()
