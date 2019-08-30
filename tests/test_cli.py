# MIT License
#
# Copyright (c) 2019 Anthony Wilder Wohns
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Test cases for the command line interface for tszip.
"""
import io
import sys
import tempfile
import pathlib
import unittest
from unittest import mock

import tskit
import msprime
import numpy as np

import tsdate 
import tsdate.cli as cli


class TestException(Exception):
    """
    Custom exception we can throw for testing.
    """


def capture_output(func, *args, **kwargs):
    """
    Runs the specified function and arguments, and returns the
    tuple (stdout, stderr) as strings.
    """
    buffer_class = io.BytesIO
    if sys.version_info[0] == 3:
        buffer_class = io.StringIO
    stdout = sys.stdout
    sys.stdout = buffer_class()
    stderr = sys.stderr
    sys.stderr = buffer_class()

    try:
        func(*args, **kwargs)
        stdout_output = sys.stdout.getvalue()
        stderr_output = sys.stderr.getvalue()
    finally:
        sys.stdout.close()
        sys.stdout = stdout
        sys.stderr.close()
        sys.stderr = stderr
    return stdout_output, stderr_output


class TestTsdateArgumentParser(unittest.TestCase):
    """
    Tests for the tsdate argument parser.
    """

    def test_default_values(self):
        parser = cli.tsdate_cli_parser()
        infile = "tmp.trees"
        args = parser.parse_args([infile])
        self.assertEqual(args.ts, [infile])
        self.assertEqual(args.clock, 'mutation')
        self.assertEqual(args.Ne, 10000)
        self.assertEqual(args.uniform, False)
        self.assertEqual(args.theta, 0.0004)
        self.assertEqual(args.rho, 0.0004)
        self.assertEqual(args.del_p, 0.02)
        self.assertEqual(args.output, "output")
        self.assertEqual(args.save_ts, False)
        self.assertEqual(args.node_dist, False) 

    def test_uniform(self):
        parser = cli.tsdate_cli_parser()
        infile = "tmp.trees"
        args = parser.parse_args([infile, '-u'])
        self.assertTrue(args.uniform)
        args = parser.parse_args([infile, "--uniform"])
        self.assertTrue(args.uniform)


class TestCli(unittest.TestCase):
    """
    Superclass of tests that run the CLI.
    """
    # Need to mock out setup_logging here or we spew logging to the console
    # in later tests.
    @mock.patch("tszip.cli.setup_logging")
    def run_tszip(self, command, mock_setup_logging):
        stdout, stderr = capture_output(cli.tszip_main, command)
        self.assertEqual(stderr, "")
        self.assertEqual(stdout, "")
        self.assertTrue(mock_setup_logging.called)

    @mock.patch("tszip.cli.setup_logging")
    def run_tsunzip(self, command, mock_setup_logging):
        stdout, stderr = capture_output(cli.tsunzip_main, command)
        self.assertEqual(stderr, "")
        self.assertEqual(stdout, "")
        self.assertTrue(mock_setup_logging.called)


class TestBadFiles(TestCli):
    """
    Tests that we deal with IO errors appropriately.
    """
    def test_sys_exit(self):
        # We test for cli.exit elsewhere as it's easier, but test that sys.exit
        # is called here, so we get coverage.
        with mock.patch("sys.exit", side_effect=TestException) as mocked_exit:
            with self.assertRaises(TestException):
                self.run_tszip(["/no/such/file"])
            mocked_exit.assert_called_once()
            args = mocked_exit.call_args[0]
            self.assertEqual(len(args), 1)
            self.assertIn("Error loading", args[0])

    def test_compress_missing(self):
        with mock.patch("tszip.cli.exit", side_effect=TestException) as mocked_exit:
            with self.assertRaises(TestException):
                self.run_tszip(["/no/such/file"])
            mocked_exit.assert_called_once()
            args = mocked_exit.call_args[0]
            self.assertEqual(len(args), 1)
            self.assertTrue(args[0].startswith("Error loading"))

    def test_decompress_missing(self):
        with mock.patch("tszip.cli.exit", side_effect=TestException) as mocked_exit:
            with self.assertRaises(TestException):
                self.run_tszip(["-d", "/no/such/file.tsz"])
            mocked_exit.assert_called_once()
            args = mocked_exit.call_args[0]
            self.assertEqual(len(args), 1)
            self.assertTrue(args[0].startswith("[Errno 2] No such file or directory"))

    def test_list_missing(self):
        with mock.patch("tszip.cli.exit", side_effect=TestException) as mocked_exit:
            with self.assertRaises(TestException):
                self.run_tszip(["-l", "/no/such/file.tsz"])
            mocked_exit.assert_called_once()
            args = mocked_exit.call_args[0]
            self.assertEqual(len(args), 1)
            self.assertTrue(args[0].startswith("[Errno 2] No such file or directory"))


class TestCompressSemantics(TestCli):
    """
    Tests that the semantics of the CLI work as expected.
    """
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory(prefix="tsdate_cli_")
        self.trees_path = pathlib.Path(self.tmpdir.name) / "msprime.trees"
        self.ts = msprime.simulate(10, mutation_rate=10, random_seed=1)
        self.ts.dump(str(self.trees_path))

    def tearDown(self):
        del self.tmpdir

    def test_simple(self):
        self.assertTrue(self.trees_path.exists())
        self.run_tsdate([str(self.trees_path)])
        self.assertFalse(self.trees_path.exists())
        outpath = pathlib.Path(str(self.trees_path) + ".tsz")
        self.assertTrue(outpath.exists())
        ts = tszip.decompress(outpath)
        self.assertEqual(ts.tables, self.ts.tables)

    def test_suffix(self):
        self.assertTrue(self.trees_path.exists())
        self.run_tszip([str(self.trees_path), "-S", ".XYZasdf"])
        self.assertFalse(self.trees_path.exists())
        outpath = pathlib.Path(str(self.trees_path) + ".XYZasdf")
        self.assertTrue(outpath.exists())
        ts = tszip.decompress(outpath)
        self.assertEqual(ts.tables, self.ts.tables)

    def test_variants_only(self):
        self.assertTrue(self.trees_path.exists())
        self.run_tszip([str(self.trees_path), "--variants-only"])
        self.assertFalse(self.trees_path.exists())
        outpath = pathlib.Path(str(self.trees_path) + ".tsz")
        self.assertTrue(outpath.exists())
        ts = tszip.decompress(outpath)
        self.assertNotEqual(ts.tables, self.ts.tables)
        G1 = ts.genotype_matrix()
        G2 = self.ts.genotype_matrix()
        self.assertTrue(np.array_equal(G1, G2))

    def test_keep(self):
        self.assertTrue(self.trees_path.exists())
        self.run_tszip([str(self.trees_path), "--keep"])
        self.assertTrue(self.trees_path.exists())
        outpath = pathlib.Path(str(self.trees_path) + ".tsz")
        self.assertTrue(outpath.exists())
        ts = tszip.decompress(outpath)
        self.assertEqual(ts.tables, self.ts.tables)

    def test_overwrite(self):
        self.assertTrue(self.trees_path.exists())
        outpath = pathlib.Path(str(self.trees_path) + ".tsz")
        outpath.touch()
        self.assertTrue(self.trees_path.exists())
        self.run_tszip([str(self.trees_path), "--force"])
        self.assertFalse(self.trees_path.exists())
        self.assertTrue(outpath.exists())
        ts = tszip.decompress(outpath)
        self.assertEqual(ts.tables, self.ts.tables)

    def test_no_overwrite(self):
        self.assertTrue(self.trees_path.exists())
        outpath = pathlib.Path(str(self.trees_path) + ".tsz")
        outpath.touch()
        self.assertTrue(self.trees_path.exists())
        with mock.patch("tszip.cli.exit", side_effect=TestException) as mocked_exit:
            with self.assertRaises(TestException):
                self.run_tszip([str(self.trees_path)])
            mocked_exit.assert_called_once_with(
                "'{}' already exists; use --force to overwrite".format(outpath))

    def test_bad_file_format(self):
        self.assertTrue(self.trees_path.exists())
        with open(str(self.trees_path), "w") as f:
            f.write("xxx")
        with mock.patch("tszip.cli.exit", side_effect=TestException) as mocked_exit:
            with self.assertRaises(TestException):
                self.run_tszip([str(self.trees_path)])
            mocked_exit.assert_called_once_with(
                "Error loading '{}': File not in KAS format".format(self.trees_path))


class DecompressSemanticsMixin(object):
    """
    Tests that the decompress semantics of the CLI work as expected.
    """
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory(prefix="tszip_cli_")
        self.trees_path = pathlib.Path(self.tmpdir.name) / "msprime.trees"
        self.ts = msprime.simulate(10, mutation_rate=10, random_seed=1)
        self.compressed_path = pathlib.Path(self.tmpdir.name) / "msprime.trees.tsz"
        tszip.compress(self.ts, self.compressed_path)

    def tearDown(self):
        del self.tmpdir

    def test_simple(self):
        self.assertTrue(self.compressed_path.exists())
        self.run_decompress([str(self.compressed_path)])
        self.assertFalse(self.compressed_path.exists())
        outpath = self.trees_path
        self.assertTrue(outpath.exists())
        ts = tskit.load(str(outpath))
        self.assertEqual(ts.tables, self.ts.tables)

    def test_suffix(self):
        suffix = ".XYGsdf"
        self.compressed_path = self.compressed_path.with_suffix(suffix)
        tszip.compress(self.ts, self.compressed_path)
        self.assertTrue(self.compressed_path.exists())
        self.run_decompress([str(self.compressed_path), "-S", suffix])
        self.assertFalse(self.compressed_path.exists())
        outpath = self.trees_path
        self.assertTrue(outpath.exists())
        ts = tskit.load(str(outpath))
        self.assertEqual(ts.tables, self.ts.tables)

    def test_keep(self):
        self.assertTrue(self.compressed_path.exists())
        self.run_decompress([str(self.compressed_path), "--keep"])
        self.assertTrue(self.compressed_path.exists())
        outpath = self.trees_path
        self.assertTrue(outpath.exists())
        ts = tskit.load(str(outpath))
        self.assertEqual(ts.tables, self.ts.tables)

    def test_overwrite(self):
        self.assertTrue(self.compressed_path.exists())
        outpath = self.trees_path
        outpath.touch()
        self.run_decompress([str(self.compressed_path), "-f"])
        self.assertFalse(self.compressed_path.exists())
        self.assertTrue(outpath.exists())
        ts = tskit.load(str(outpath))
        self.assertEqual(ts.tables, self.ts.tables)

    def test_no_overwrite(self):
        self.assertTrue(self.compressed_path.exists())
        outpath = self.trees_path
        outpath.touch()
        with mock.patch("tszip.cli.exit", side_effect=TestException) as mocked_exit:
            with self.assertRaises(TestException):
                self.run_decompress([str(self.compressed_path)])
            mocked_exit.assert_called_once_with(
                "'{}' already exists; use --force to overwrite".format(outpath))

    def test_decompress_bad_suffix(self):
        with mock.patch("tszip.cli.exit", side_effect=TestException) as mocked_exit:
            with self.assertRaises(TestException):
                self.run_decompress([str(self.compressed_path), "-S", "asdf"])
            mocked_exit.assert_called_once_with(
                "Compressed file must have 'asdf' suffix")

    def test_bad_file_format(self):
        self.assertTrue(self.compressed_path.exists())
        with open(str(self.compressed_path), "w") as f:
            f.write("xxx")
        with mock.patch("tszip.cli.exit", side_effect=TestException) as mocked_exit:
            with self.assertRaises(TestException):
                self.run_decompress([str(self.compressed_path)])
            mocked_exit.assert_called_once_with(
                "Error reading '{}': File is not in tgzip format".format(
                    self.compressed_path))


class TestDecompressSemanticsTszip(DecompressSemanticsMixin, TestCli):
    def run_decompress(self, args):
        self.run_tszip(["-d"] + args)


class TestDecompressSemanticsTsunzip(DecompressSemanticsMixin, TestCli):
    def run_decompress(self, args):
        self.run_tsunzip(args)


class TestList(unittest.TestCase):
    """
    Tests that the --list option works as expected.

    We don't need to mock out setup_logging here because it's not called for list.
    """
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory(prefix="tszip_cli_")
        self.trees_path = pathlib.Path(self.tmpdir.name) / "msprime.trees"
        self.ts = msprime.simulate(10, mutation_rate=10, random_seed=1)
        self.compressed_path = pathlib.Path(self.tmpdir.name) / "msprime.trees.tsz"
        tszip.compress(self.ts, self.compressed_path)

    def tearDown(self):
        del self.tmpdir

    def test_simple(self):
        stdout, stderr = capture_output(
            cli.tszip_main, ["--list", str(self.compressed_path)])
        self.assertEqual(stderr, "")
        lines = stdout.splitlines()
        self.assertTrue(lines[0].startswith("File: {}".format(self.compressed_path)))
        for line in lines:
            self.assertGreater(len(line), 0)

    def test_verbose(self):
        stdout, stderr = capture_output(
            cli.tszip_main, ["--list", "-v", str(self.compressed_path)])
        self.assertEqual(stderr, "")
        lines = stdout.splitlines()
        self.assertTrue(lines[0].startswith("File: {}".format(self.compressed_path)))
        for line in lines:
            self.assertGreater(len(line), 0)

    def test_bad_file_format(self):
        self.assertTrue(self.compressed_path.exists())
        with open(str(self.compressed_path), "w") as f:
            f.write("xxx")
        with mock.patch("tszip.cli.exit", side_effect=TestException) as mocked_exit:
            with self.assertRaises(TestException):
                cli.tszip_main([str(self.compressed_path), "-l"])
            mocked_exit.assert_called_once_with(
                "Error reading '{}': File is not in tgzip format".format(
                    self.compressed_path))


class TestSetupLogging(unittest.TestCase):
    """
    Tests that setup logging has the desired effect.
    """
    def test_default(self):
        parser = cli.tszip_cli_parser()
        args = parser.parse_args(["afile"])
        with mock.patch("logging.basicConfig") as mocked_setup:
            cli.setup_logging(args)
            mocked_setup.assert_called_once_with(level="WARN", format=cli.log_format)

    def test_verbose(self):
        parser = cli.tszip_cli_parser()
        args = parser.parse_args(["afile", "-v"])
        with mock.patch("logging.basicConfig") as mocked_setup:
            cli.setup_logging(args)
            mocked_setup.assert_called_once_with(level="INFO", format=cli.log_format)

    def test_very_verbose(self):
        parser = cli.tszip_cli_parser()
        args = parser.parse_args(["afile", "-vv"])
        with mock.patch("logging.basicConfig") as mocked_setup:
            cli.setup_logging(args)
            mocked_setup.assert_called_once_with(level="DEBUG", format=cli.log_format)
