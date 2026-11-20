"""
Parity of ``LogLikelihoods.logsumexp`` (a streaming log-sum-exp) with its oracle,
including NaN and infinities (which are not errors) and non-contiguous input.
"""

import numpy as np
import pytest
from oracle import discrete
from oracle.helpers import INF, NAN, assert_outcome, assert_ulp_close, compare

# x, outcome category of the oracle
EDGE_CASES = [
    pytest.param([], "nonfinite", id="empty"),
    pytest.param([5.0], "finite", id="single"),
    pytest.param([1e-320], "finite", id="single_subnormal"),
    pytest.param([-INF], "nonfinite", id="single_minus_inf"),
    pytest.param([-INF] * 7, "nonfinite", id="all_minus_inf"),
    pytest.param([-INF, 5.0, -INF], "finite", id="minus_inf_ignored"),
    pytest.param([-3.5, -INF, -INF], "finite", id="finite_first"),
    pytest.param([-INF, -INF, -3.5], "finite", id="finite_last"),
    pytest.param([-0.0, 0.0], "finite", id="negative_zero"),
    pytest.param([1000.0, 1000.0], "finite", id="would_overflow_naively"),
    pytest.param([-1000.0, -1000.0], "finite", id="would_underflow_naively"),
    pytest.param([1e308, 1e308], "finite", id="huge_equal"),
    pytest.param([-1e308, 1e308], "finite", id="huge_opposite"),
    pytest.param([1e308, -1e308], "finite", id="huge_opposite_reversed"),
    pytest.param([-800.0, 0.0, -800.0], "finite", id="dominated_middle"),
    pytest.param([0.0, -800.0, -800.0], "finite", id="dominated_first"),
    pytest.param([-800.0, -800.0, 0.0], "finite", id="dominated_last"),
    pytest.param([1.0, 2.0, 3.0], "finite", id="increasing"),
    pytest.param([3.0, 2.0, 1.0], "finite", id="decreasing"),
    # NOTE: a single +inf gives +inf, but inf - inf makes NaN with two of them
    pytest.param([INF], "nonfinite", id="inf"),
    pytest.param([0.0, INF], "nonfinite", id="finite_and_inf"),
    pytest.param([INF, INF], "nan", id="two_inf"),
    pytest.param([INF, -INF, INF], "nan", id="two_inf_with_minus_inf"),
    pytest.param([INF, -INF], "nonfinite", id="inf_and_minus_inf"),
    # NOTE: NaN is not -inf and fails `x <= alpha`, so it takes the "new maximum"
    # branch and poisons the result wherever it is
    pytest.param([NAN], "nan", id="nan"),
    pytest.param([NAN, 1.0], "nan", id="nan_first"),
    pytest.param([1.0, NAN], "nan", id="nan_last"),
    pytest.param([1.0, NAN, 2.0], "nan", id="nan_middle"),
]


class TestLogsumexp:
    @pytest.mark.parametrize(("x", "expected_kind"), EDGE_CASES)
    def test_edge_cases(self, impl, x, expected_kind):
        x = np.array(x, dtype=float)
        assert_outcome(impl.discrete.logsumexp, discrete.logsumexp, (x,), expected_kind)

    def test_non_contiguous_input(self, impl):
        # `compare` copies its arguments to contiguous arrays, so call directly
        x = np.arange(1.0, 10.0)[::2]
        assert not x.flags.c_contiguous
        assert_ulp_close(impl.discrete.logsumexp(x), discrete.logsumexp(x))

    @pytest.mark.parametrize("mode", ["typical", "extreme"])
    @pytest.mark.parametrize("seed", range(4))
    def test_random(self, impl, seed, mode):
        rng = np.random.default_rng(seed)
        size = int(rng.integers(1, 500))
        scale = {"typical": 5.0, "extreme": 1e5}[mode]
        x = rng.normal(0, scale, size)
        x[rng.random(size) < rng.uniform(0, 1)] = -INF
        compare(impl.discrete.logsumexp, discrete.logsumexp, x)
