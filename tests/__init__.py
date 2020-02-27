"""
Package definition for tests. Defined to allow cross-importing.
"""
import unittest
import tempfile

import tsdate


class CacheReadingTest(unittest.TestCase):
    """
    This should be used as the superclass of all tests that use precalculated data.
    Rather than using the standard userdir for precalculated data, we use this
    local directory that can be easily removed and modified without fear of
    interfering with production code.
    """
    cache_dir = "test_cache"
    saved_cache_dir = None
    saved_urls = {}

    def setUp(self):
        self.saved_cache_dir = tsdate.get_cache_dir()
        tsdate.set_cache_dir(self.cache_dir)

    def tearDown(self):
        tsdate.set_cache_dir(self.saved_cache_dir)


class CacheWritingTest(unittest.TestCase):
    """
    This should be used as the superclass of all tests that alter the
    precalculated data cache in any non-standard way.
    """
    saved_cache_dir = None
    tmp_cache_dir = None

    def setUp(self):
        self.saved_cache_dir = tsdate.get_cache_dir()
        self.tmp_cache_dir = tempfile.TemporaryDirectory()
        tsdate.set_cache_dir(self.tmp_cache_dir.name)

    def tearDown(self):
        tsdate.set_cache_dir(self.saved_cache_dir)
        del self.tmp_cache_dir
