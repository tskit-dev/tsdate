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
import tempfile  # NOQA - not currently used
import pathlib  # NOQA - not currently used
import unittest
from unittest import mock  # NOQA - not currently used

import tskit  # NOQA - not currently used
import msprime  # NOQA - not currently used
import numpy as np  # NOQA - not currently used

import tsdate  # NOQA - not currently used
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
        stdout = sys.stdout.getvalue()
        stderr = sys.stderr.getvalue()
    finally:
        sys.stdout.close()
        sys.stdout = stdout
        sys.stderr.close()
        sys.stderr = stderr
    return stdout, stderr


class TestTsdateArgParser(unittest.TestCase):
    """
    Tests for the tsdate argument parser.
    """
    infile = "tmp.trees"
    output = "output.trees"

    def test_default_values(self):
        parser = cli.tsdate_cli_parser()
        args = parser.parse_args([self.infile, self.output])
        self.assertEqual(args.ts, self.infile)
        self.assertEqual(args.output, self.output)
        self.assertEqual(args.Ne, 10000)
        self.assertEqual(args.time_grid, "adaptive")
        self.assertEqual(args.mutation_rate, None)
        self.assertEqual(args.recombination_rate, None)
        self.assertEqual(args.slices, 50)
        self.assertEqual(args.epsilon, 1e-6)
        self.assertEqual(args.num_threads, 0)
        self.assertFalse(args.progress)

    def test_Ne(self):
        parser = cli.tsdate_cli_parser()
        args = parser.parse_args([self.infile, self.output, "-n", "10000"])
        self.assertEqual(args.Ne, 10000)
        args = parser.parse_args([self.infile, self.output, "--Ne", "10000"])
        self.assertEqual(args.Ne, 10000)

    def test_time_grid(self):
        parser = cli.tsdate_cli_parser()
        args = parser.parse_args(
            [self.infile, self.output, "-g", "adaptive"])
        self.assertEqual(args.time_grid, "adaptive")
        args = parser.parse_args([self.infile, self.output, "--time-grid", "uniform"])
        self.assertEqual(args.time_grid, "uniform")

    def test_mutation_rate(self):
        parser = cli.tsdate_cli_parser()
        args = parser.parse_args([self.infile, self.output, "-m", "1e10"])
        self.assertEqual(args.mutation_rate, 1e10)
        args = parser.parse_args([self.infile, self.output, "--mutation-rate", "1e10"])
        self.assertEqual(args.mutation_rate, 1e10)

    def test_recombination_rate(self):
        parser = cli.tsdate_cli_parser()
        args = parser.parse_args([self.infile, self.output, "-r", "1e-100"])
        self.assertEqual(args.recombination_rate, 1e-100)
        args = parser.parse_args(
            [self.infile, self.output, "--recombination-rate", "1e-100"])
        self.assertEqual(args.recombination_rate, 1e-100)

    def test_slices(self):
        parser = cli.tsdate_cli_parser()
        args = parser.parse_args([self.infile, self.output, "-s", "100"])
        self.assertEqual(args.slices, 100)
        args = parser.parse_args([self.infile, self.output, "--slices", "100"])
        self.assertEqual(args.slices, 100)

    def test_epsilon(self):
        parser = cli.tsdate_cli_parser()
        args = parser.parse_args([self.infile, self.output, "-e", "123"])
        self.assertEqual(args.epsilon, 123)
        args = parser.parse_args([self.infile, self.output, "--epsilon", "321"])
        self.assertEqual(args.epsilon, 321)


class TestCli(unittest.TestCase):
    """
    Superclass of tests that run the CLI.
    """
    def run_tsdate(self, command):
        stdout, stderr = capture_output(cli.tsdate_main, command)
        self.assertEqual(stderr, "")
        self.assertEqual(stdout, "")
