"""
Parity of ``_marginalize_over_ancestors`` and ``conditional_coalescent_variance``
with their oracles, and a pin of how ``tsdate`` calls the latter with a single tip.
"""

import numpy as np
import pytest
import tskit
from oracle import prior
from oracle.helpers import INF, NAN, assert_outcome, compare

import tsdate
from tsdate import prior as tsdate_prior


def coalescent_moments(num_tips):
    """The argument that `conditional_coalescent_variance` passes"""
    rates = np.array(
        [2 / (i * (i - 1)) if i > 1 else 0.0 for i in range(1, num_tips + 1)]
    )
    mean, variance = rates.copy(), rates.copy() ** 2
    for i in range(rates.size - 2, 0, -1):
        mean[i] += mean[i + 1]
        variance[i] += variance[i + 1]
    return np.stack((mean, variance + mean**2), 1)


# val, outcome category of the oracle
MARGINALIZE_EDGE_CASES = [
    pytest.param(np.array([[1.0, 2.0], [3.0, 4.0]]), "finite", id="two_tips"),
    pytest.param(np.zeros((6, 2)), "finite", id="zeros"),
    pytest.param(np.full((6, 2), NAN), "nan", id="all_nan"),
    # NOTE: non-finite values are propagated by the weighted sum, with no checks
    pytest.param(
        np.array([[1.0, 2.0], [INF, 1.0], [3.0, 3.0], [1.0, NAN]]), "nan", id="inf_nan"
    ),
]


class TestMarginalizeOverAncestors:
    @pytest.mark.parametrize("num_tips", [2, 3, 10, 60, 200])
    @pytest.mark.parametrize("seed", range(2))
    def test_random(self, impl, seed, num_tips):
        rng = np.random.default_rng(seed)
        val = rng.uniform(0.1, 10.0, (num_tips, 2))
        result = compare(
            impl.prior._marginalize_over_ancestors,
            prior._marginalize_over_ancestors,
            val,
        )
        assert result.value.shape == (num_tips + 1, 2)

    @pytest.mark.parametrize("num_columns", [1, 3])
    def test_number_of_moments(self, impl, num_columns):
        val = np.random.default_rng(0).uniform(0.1, 10.0, (12, num_columns))
        compare(
            impl.prior._marginalize_over_ancestors,
            prior._marginalize_over_ancestors,
            val,
        )

    @pytest.mark.parametrize("num_tips", [3, 10, 100])
    def test_coalescent_moments(self, impl, num_tips):
        compare(
            impl.prior._marginalize_over_ancestors,
            prior._marginalize_over_ancestors,
            coalescent_moments(num_tips),
        )

    @pytest.mark.parametrize(("val", "expected_kind"), MARGINALIZE_EDGE_CASES)
    def test_edge_cases(self, impl, val, expected_kind):
        assert_outcome(
            impl.prior._marginalize_over_ancestors,
            prior._marginalize_over_ancestors,
            (val,),
            expected_kind,
        )

    def test_input_not_modified(self, impl):
        # `compare` passes copies, so call directly
        val = np.random.default_rng(0).uniform(0.1, 10.0, (8, 2))
        copy = val.copy()
        impl.prior._marginalize_over_ancestors(val)
        assert np.array_equal(val, copy)


class TestConditionalCoalescentVariance:
    @pytest.mark.parametrize("num_tips", [2, 3, 5, 20, 100, 250])
    def test_variance(self, impl, num_tips):
        result = compare(
            impl.prior.conditional_coalescent_variance,
            prior.conditional_coalescent_variance,
            num_tips,
        )
        assert result.value.shape == (num_tips + 1,)


class TestSingleTip:
    def test_called_with_one_tip(self, monkeypatch):
        # NOTE: current behaviour, possibly a bug: a tree with one
        # attached sample (with `allow_unary`) calls
        # `conditional_coalescent_variance(1)`, whose result is unused. That call is
        # replaced here, because in numba it reads and writes out of bounds.
        tables = tskit.TableCollection(1000.0)
        for _ in range(3):
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.nodes.add_row(time=1.0)
        tables.edges.add_row(0, 1000, 3, 0)
        tables.edges.add_row(500, 1000, 3, 1)
        tables.edges.add_row(500, 1000, 3, 2)
        tables.sort()
        ts = tables.tree_sequence()
        called = []
        original = tsdate_prior.conditional_coalescent_variance

        def record(num_tips):
            called.append(int(num_tips))
            if num_tips == 1:
                return np.zeros(2)
            return original(num_tips)

        monkeypatch.setattr(tsdate_prior, "conditional_coalescent_variance", record)
        tsdate.build_prior_grid(ts, timepoints=5, population_size=1e4, allow_unary=True)
        assert called == [3, 1]
