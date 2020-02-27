"""
Tests for the cache management code.
"""
import pathlib
import os

import appdirs
import numpy as np
import msprime

import tsdate
from tsdate import cache
import tests


class TestSetCacheDir(tests.CacheWritingTest):
    """
    Tests the set_cache_dir function.
    """
    paths = [
        "/somefile", "/some/other/file/", "relative/path", "relative/path/"]

    def test_paths(self):
        for test in self.paths:
            tsdate.set_cache_dir(test)
            self.assertEqual(tsdate.get_cache_dir(), pathlib.Path(test))
            tsdate.set_cache_dir(pathlib.Path(test))
            self.assertEqual(tsdate.get_cache_dir(), pathlib.Path(test))

    def test_none(self):
        tsdate.set_cache_dir(None)
        cache_dir = pathlib.Path(appdirs.user_cache_dir("tsdate", "tsdate"))
        self.assertEqual(tsdate.get_cache_dir(), cache_dir)


class TestReadingCacheDir(tests.CacheReadingTest):
    """
    Tests we can get cache dir and read from it correctly
    """
    def test_prior_writes_to_default_cache(self):
        ts = msprime.simulate(10)
        tsdate.build_prior_grid(ts, timepoints=20, approximate_prior=True, approx_prior_size=100)
        cache_dir = cache.get_cache_dir()
        precalc_file = os.path.join(cache_dir, "prior_100df_{}.txt".format(tsdate.__version__))
        self.assertTrue(os.path.exists(precalc_file)) 
        approx_prior = np.genfromtxt(precalc_file)
        self.assertTrue(approx_prior.shape == (100, 2))

