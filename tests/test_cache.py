"""
Tests for the cache management code.
"""
import unittest
import pathlib
import os

import appdirs
import numpy as np
import msprime

import tsdate
from tsdate.date import ConditionalCoalescentTimes
from tsdate import cache
import tests


class TestSetCacheDir(unittest.TestCase):
    """
    Tests the set_cache_dir function.
    """

    def test_cache_dir_exists(self):
        tsdate.set_cache_dir()
        cache_dir = pathlib.Path(appdirs.user_cache_dir("tsdate", "tsdate"))
        self.assertEqual(tsdate.get_cache_dir(), cache_dir)

    def test_cached_prior(self):
        """
        Tests the caching of the precalculated prior
        """
        # Force approx prior with a tiny n
        priors_approx10 = ConditionalCoalescentTimes(10)
        priors_approx10.add(10)
        # Check we have created the prior file
        self.assertTrue(
            os.path.isfile(ConditionalCoalescentTimes.get_precalc_cache(10)))
        priors_approxNone = ConditionalCoalescentTimes(None)
        priors_approxNone.add(10)
        self.assertTrue(
            np.allclose(priors_approx10[10], priors_approxNone[10], equal_nan=True))
        # Test when using a bigger n that we're using the precalculated version
        priors_approx10.add(100)
        self.assertEquals(priors_approx10[100].shape[0], 100 + 1)
        priors_approxNone.add(100, approximate=False)
        self.assertEquals(priors_approxNone[100].shape[0], 100 + 1)
        self.assertFalse(
            np.allclose(priors_approx10[100], priors_approxNone[100], equal_nan=True))

        priors_approx10.clear_precalculated_prior()
        self.assertFalse(
            os.path.isfile(ConditionalCoalescentTimes.get_precalc_cache(10)),
            "The file `{}` should have been deleted, but has not been.\
             Please delete it")
