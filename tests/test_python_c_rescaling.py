"""
Tests of the rescaling functions of the low-level _tsdate module that the oracle tests
in test_oracle_rescaling.py do not reach: argument handling, and inputs outside what
a tree sequence can produce. Values are compared with the oracles.
"""

import _tsdate
import numpy as np
import pytest
from oracle import rescaling
from oracle.helpers import compare

NODES_TIME = np.array([0.0, 0.0, 1.0, 2.0])
PARENTS = np.array([2, 3], dtype=np.int32)
CHILDREN = np.array([0, 2], dtype=np.int32)


class TestArguments:
    def test_epochs_beyond_int32(self):
        with pytest.raises(OverflowError):
            _tsdate.fixed_changepoints(np.ones(3), 2**70)

    def test_wrong_dtype(self):
        with pytest.raises(TypeError):
            _tsdate.mutational_area(
                NODES_TIME, np.ones((2, 2)), PARENTS.astype(np.int64), CHILDREN
            )

    def test_wrong_dimension(self):
        with pytest.raises(ValueError, match="depth"):
            _tsdate.mutational_area(NODES_TIME, np.ones(4), PARENTS, CHILDREN)

    def test_empty_breaks(self):
        empty = np.zeros(0)
        with pytest.raises(ValueError, match="non-empty"):
            _tsdate.piecewise_scale_point_estimate(
                np.zeros(1), np.zeros(1, dtype=bool), empty, empty
            )
        with pytest.raises(ValueError, match="non-empty"):
            _tsdate.piecewise_scale_posterior(
                np.ones((1, 2)), np.zeros(1, dtype=bool), empty, empty, 0.5, 1000.0
            )

    def test_point_estimate_fixed_size(self):
        with pytest.raises(ValueError, match="same size"):
            _tsdate.piecewise_scale_point_estimate(
                np.zeros(2), np.zeros(1, dtype=bool), np.zeros(1), np.zeros(1)
            )

    def test_posterior_columns(self):
        with pytest.raises(ValueError, match="two columns"):
            _tsdate.piecewise_scale_posterior(
                np.ones((1, 3)),
                np.zeros(1, dtype=bool),
                np.zeros(1),
                np.zeros(1),
                0.5,
                1.0,
            )

    def test_poisson_sizes(self):
        with pytest.raises(ValueError, match="same size"):
            _tsdate.poisson_changepoints(np.ones(3), np.ones(2), 2.0, 0.0, 0.0)

    def test_count_mutations_sizes(self):
        args = [
            np.zeros(2, dtype=bool),
            np.zeros(1, dtype=np.int32),
            np.zeros(1),
            np.zeros(1, dtype=np.int32),
            np.zeros(1, dtype=np.int32),
            np.zeros(1),
            np.zeros(1),
            np.zeros(1, dtype=np.int32),
            np.zeros(1, dtype=np.int32),
            1.0,
            False,
        ]
        for index in [4, 7, 2]:
            bad = list(args)
            bad[index] = np.zeros(2, dtype=bad[index].dtype)
            with pytest.raises(ValueError, match="same size|one entry per edge"):
                _tsdate.count_mutations(*bad)


class TestOutsideTreeSequences:
    def test_nan_node_times(self):
        # NaN is ordered last, whichever its position
        times = np.array([0.0, np.nan, 1.0, np.nan, 2.0])
        likelihoods = np.array([[1.0, 1.0], [1.0, 1.0]])
        parents = np.array([2, 4], dtype=np.int32)
        children = np.array([0, 2], dtype=np.int32)
        compare(
            _tsdate.mutational_area,
            rescaling.mutational_area,
            times,
            likelihoods,
            parents,
            children,
        )

    def test_changepoint_before_the_first_epoch(self):
        # The mutational spans make the offsets of the four epochs 1, 1, -3 and 1, so
        # that the total weight is zero and the cumulative distribution is NaN where
        # it is looked up: the first changepoint is -1, which counts from the end as
        # in numpy.
        nodes_time = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0])
        likelihoods = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, -3.0], [1.0, 1.0]])
        parents = np.array([2, 3, 4, 5], dtype=np.int32)
        children = np.array([0, 2, 3, 4], dtype=np.int32)
        fixed = np.full(6, False)
        args = (nodes_time, likelihoods, fixed, parents, children, 2)
        result = compare(
            _tsdate.mutational_timescale, rescaling.mutational_timescale, *args
        )
        assert result.value[0].tolist() == [4.0, 4.0]

    def test_quantile_below_first_breakpoint(self):
        # the lower quantile (1.1) is below the first breakpoint, the mean (2) is not
        args = (
            np.array([[1.0, 1.0]]),
            np.array([False]),
            np.array([1.5, 3.0, 5.0]),
            np.array([0.0, 1.0, 2.0]),
            0.5,
            1000.0,
        )
        result = compare(
            _tsdate.piecewise_scale_posterior, rescaling.piecewise_scale_posterior, *args
        )
        assert isinstance(result.exception, AssertionError)

    def test_fixed_posteriors_below_first_breakpoint(self):
        # the rescaled quantiles of fixed posteriors are zero, and are checked too
        args = (
            np.array([[1.0, 1.0]]),
            np.array([True]),
            np.array([0.5, 2.0, 5.0]),
            np.array([0.0, 1.0, 2.0]),
            0.5,
            1000.0,
        )
        result = compare(
            _tsdate.piecewise_scale_posterior, rescaling.piecewise_scale_posterior, *args
        )
        assert isinstance(result.exception, AssertionError)
