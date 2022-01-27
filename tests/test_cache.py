"""
Tests for the cache management code.
"""
import os
import pathlib
import unittest

import appdirs
import numpy as np

import tsdate
from tsdate.prior import ConditionalCoalescentTimes


class TestSetCacheDir(unittest.TestCase):
    """
    Tests the set_cache_dir function.
    """

    def test_cache_dir_exists(self):
        cache_dir = pathlib.Path(appdirs.user_cache_dir("tsdate", "tsdate"))
        assert tsdate.get_cache_dir() == cache_dir

    def test_cached_prior(self):
        # Force approx prior with a tiny n
        fn = ConditionalCoalescentTimes.get_precalc_cache(10)
        if os.path.isfile(fn):
            raise AssertionError(f"The file {fn} already exists. Delete before testing")
        with self.assertLogs(level="WARNING") as log:
            priors_approx10 = ConditionalCoalescentTimes(10, Ne=1)
            assert len(log.output) == 1
            assert "user cache" in log.output[0]
        priors_approx10.add(10)
        # Check we have created the prior file
        assert os.path.isfile(fn)
        priors_approxNone = ConditionalCoalescentTimes(None, Ne=1)
        priors_approxNone.add(10)
        assert np.allclose(priors_approx10[10], priors_approxNone[10], equal_nan=True)
        # Test when using a bigger n that we're using the precalculated version
        priors_approx10.add(100)
        assert priors_approx10[100].shape[0] == 100 + 1
        priors_approxNone.add(100, approximate=False)
        assert priors_approxNone[100].shape[0] == 100 + 1
        assert not np.allclose(
            priors_approx10[100], priors_approxNone[100], equal_nan=True
        )

        priors_approx10.clear_precalculated_priors()
        assert not os.path.isfile(fn), (
            "The file "
            + fn
            + "should have been "
            + "deleted, but has not been. Please delete it"
        )
