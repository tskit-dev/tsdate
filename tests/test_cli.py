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
Test cases for the command line interface for tsdate.
"""
import io
import sys
import unittest
from unittest import mock

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
        args = parser.parse_args(infile)
        self.assertEqual(args.ts, infile)
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
    def run_tsdate(self, command):
        stdout, stderr = capture_output(cli.tsdate_main, command)
        self.assertEqual(stderr, "")
        self.assertEqual(stdout, "")


class TestBadFiles(TestCli):
    """
    Tests that we deal with IO errors appropriately.
    """
    def test_sys_exit(self):
        # We test for cli.exit elsewhere as it's easier, but test that sys.exit
        # is called here, so we get coverage.
        with mock.patch("sys.exit", side_effect=TestException) as mocked_exit:
            with self.assertRaises(TestException):
                self.run_tsdate(["/no/such/file"])
            mocked_exit.assert_called_once()
            args = mocked_exit.call_args[0]
            self.assertEqual(len(args), 1)
            self.assertIn("Error loading", args[0])


class TestSetupLogging(unittest.TestCase):
    """
    Tests that setup logging has the desired effect.
    """
    def test_default(self):
        parser = cli.tsdate_cli_parser()
        args = parser.parse_args(["afile"])
        with mock.patch("logging.basicConfig") as mocked_setup:
            cli.setup_logging(args)
            mocked_setup.assert_called_once_with(level="WARN", format=cli.log_format)

    def test_verbose(self):
        parser = cli.tsdate_cli_parser()
        args = parser.parse_args(["afile", "-v"])
        with mock.patch("logging.basicConfig") as mocked_setup:
            cli.setup_logging(args)
            mocked_setup.assert_called_once_with(level="INFO", format=cli.log_format)

    def test_very_verbose(self):
        parser = cli.tsdate_cli_parser()
        args = parser.parse_args(["afile", "-vv"])
        with mock.patch("logging.basicConfig") as mocked_setup:
            cli.setup_logging(args)
            mocked_setup.assert_called_once_with(level="DEBUG", format=cli.log_format)
