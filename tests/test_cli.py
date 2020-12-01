# MIT License
#
# Copyright (c) 2020 University of Oxford
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
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import msprime
import numpy as np
import tskit

import tsdate
import tsdate.cli as cli


class TestTsdateArgParser(unittest.TestCase):
    """
    Tests for the tsdate argument parser.
    """

    infile = "tmp.trees"
    output = "output.trees"

    def test_default_values(self):
        with mock.patch("tsdate.cli.setup_logging"):
            parser = cli.tsdate_cli_parser()
            args = parser.parse_args(["date", self.infile, self.output, "1"])
        self.assertEqual(args.tree_sequence, self.infile)
        self.assertEqual(args.output, self.output)
        self.assertEqual(args.Ne, 1)
        self.assertEqual(args.mutation_rate, None)
        self.assertEqual(args.recombination_rate, None)
        self.assertEqual(args.epsilon, 1e-6)
        self.assertEqual(args.num_threads, None)
        self.assertEqual(args.probability_space, "logarithmic")
        self.assertEqual(args.method, "inside_outside")
        self.assertFalse(args.progress)

    def test_mutation_rate(self):
        parser = cli.tsdate_cli_parser()
        args = parser.parse_args(
            ["date", self.infile, self.output, "10000", "-m", "1e10"]
        )
        self.assertEqual(args.mutation_rate, 1e10)
        args = parser.parse_args(
            ["date", self.infile, self.output, "10000", "--mutation-rate", "1e10"]
        )
        self.assertEqual(args.mutation_rate, 1e10)

    def test_recombination_rate(self):
        parser = cli.tsdate_cli_parser()
        args = parser.parse_args(
            ["date", self.infile, self.output, "10000", "-r", "1e-100"]
        )
        self.assertEqual(args.recombination_rate, 1e-100)
        args = parser.parse_args(
            [
                "date",
                self.infile,
                self.output,
                "10000",
                "--recombination-rate",
                "1e-100",
            ]
        )
        self.assertEqual(args.recombination_rate, 1e-100)

    def test_epsilon(self):
        parser = cli.tsdate_cli_parser()
        args = parser.parse_args(
            ["date", self.infile, self.output, "10000", "-e", "123"]
        )
        self.assertEqual(args.epsilon, 123)
        args = parser.parse_args(
            ["date", self.infile, self.output, "10000", "--epsilon", "321"]
        )
        self.assertEqual(args.epsilon, 321)

    def test_num_threads(self):
        parser = cli.tsdate_cli_parser()
        args = parser.parse_args(
            ["date", self.infile, self.output, "10000", "--num-threads", "1"]
        )
        self.assertEqual(args.num_threads, 1)
        args = parser.parse_args(
            ["date", self.infile, self.output, "10000", "--num-threads", "2"]
        )
        self.assertEqual(args.num_threads, 2)

    def test_probability_space(self):
        parser = cli.tsdate_cli_parser()
        args = parser.parse_args(
            ["date", self.infile, self.output, "10000", "--probability-space", "linear"]
        )
        self.assertEqual(args.probability_space, "linear")
        args = parser.parse_args(
            [
                "date",
                self.infile,
                self.output,
                "10000",
                "--probability-space",
                "logarithmic",
            ]
        )
        self.assertEqual(args.probability_space, "logarithmic")

    def test_method(self):
        parser = cli.tsdate_cli_parser()
        args = parser.parse_args(
            ["date", self.infile, self.output, "10000", "--method", "inside_outside"]
        )
        self.assertEqual(args.method, "inside_outside")
        args = parser.parse_args(
            ["date", self.infile, self.output, "10000", "--method", "maximization"]
        )
        self.assertEqual(args.method, "maximization")

    def test_progress(self):
        parser = cli.tsdate_cli_parser()
        args = parser.parse_args(
            ["date", self.infile, self.output, "10000", "--progress"]
        )
        self.assertTrue(args.progress)

    def test_default_values_preprocess(self):
        with mock.patch("tsdate.cli.setup_logging"):
            parser = cli.tsdate_cli_parser()
            args = parser.parse_args(["preprocess", self.infile, self.output])
        self.assertEqual(args.tree_sequence, self.infile)
        self.assertEqual(args.output, self.output)
        self.assertEqual(args.minimum_gap, 1000000)
        self.assertTrue(args.trim_telomeres)


class TestEndToEnd(unittest.TestCase):
    """
    Class to test input to CLI outputs dated tree sequences.
    """

    def ts_equal(self, ts1, ts2, times_equal=False):
        self.assertEqual(ts1.sequence_length, ts2.sequence_length)
        t1 = ts1.tables
        t2 = ts2.tables
        self.assertEqual(t1.sites, t2.sites)
        self.assertEqual(t1.mutations, t2.mutations)
        # Edges may have been re-ordered, since sortedness requirements specify
        # they are sorted by parent time, and the relative order of
        # (unconnected) parent nodes might have changed due to time inference
        self.assertEquals(set(t1.edges), set(t2.edges))
        if not times_equal:
            # The dated and undated tree sequences should not have the same node times
            self.assertTrue(
                not np.array_equal(ts1.tables.nodes.time, ts2.tables.nodes.time)
            )
            # New tree sequence will have node times in metadata
            for column_name in t1.nodes.column_names:
                if column_name not in ["time", "metadata", "metadata_offset"]:
                    col_t1 = getattr(t1.nodes, column_name)
                    col_t2 = getattr(t2.nodes, column_name)
                    self.assertTrue(np.array_equal(col_t1, col_t2))
            # Assert that last provenance shows tree sequence was dated
            self.assertEqual(len(t1.provenances), len(t2.provenances) - 1)
            for index, (prov1, prov2) in enumerate(zip(t1.provenances, t2.provenances)):
                self.assertEqual(prov1, prov2)
                if index == len(t1.provenances) - 1:
                    break
            self.assertEqual(
                json.loads(t2.provenances[-1].record)["software"]["name"], "tsdate"
            )

        else:
            self.assertEqual(t1.nodes, t2.nodes)

    def verify(self, input_ts, cmd):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_filename = pathlib.Path(tmpdir) / "input.trees"
            input_ts.dump(input_filename)
            output_filename = pathlib.Path(tmpdir) / "output.trees"
            full_cmd = "date " + str(input_filename) + f" {output_filename} " + cmd
            cli.tsdate_main(full_cmd.split())
            output_ts = tskit.load(output_filename)
        self.assertEqual(input_ts.num_samples, output_ts.num_samples)
        self.ts_equal(input_ts, output_ts)

    def compare_python_api(self, input_ts, cmd, Ne, mutation_rate, method):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_filename = pathlib.Path(tmpdir) / "input.trees"
            input_ts.dump(input_filename)
            output_filename = pathlib.Path(tmpdir) / "output.trees"
            full_cmd = "date " + str(input_filename) + f" {output_filename} " + cmd
            cli.tsdate_main(full_cmd.split())
            output_ts = tskit.load(output_filename)
        dated_ts = tsdate.date(
            input_ts, Ne=Ne, mutation_rate=mutation_rate, method=method
        )
        print(dated_ts.tables.nodes.time, output_ts.tables.nodes.time)
        self.assertTrue(
            np.array_equal(dated_ts.tables.nodes.time, output_ts.tables.nodes.time)
        )

    def test_ts(self):
        input_ts = msprime.simulate(10, random_seed=1)
        cmd = "1"
        self.verify(input_ts, cmd)

    def test_mutation_rate(self):
        input_ts = msprime.simulate(10, random_seed=1)
        cmd = "1 --mutation-rate 1e-8"
        self.verify(input_ts, cmd)

    def test_recombination_rate(self):
        input_ts = msprime.simulate(10, random_seed=1)
        cmd = "1 --recombination-rate 1e-8"
        self.assertRaises(NotImplementedError, self.verify, input_ts, cmd)

    def test_epsilon(self):
        input_ts = msprime.simulate(10, random_seed=1)
        cmd = "1 --epsilon 1e-3"
        self.verify(input_ts, cmd)

    def test_num_threads(self):
        input_ts = msprime.simulate(10, random_seed=1)
        cmd = "1 --num-threads 2"
        self.verify(input_ts, cmd)

    def test_probability_space(self):
        input_ts = msprime.simulate(10, random_seed=1)
        cmd = "1 --probability-space linear"
        self.verify(input_ts, cmd)
        cmd = "1 --probability-space logarithmic"
        self.verify(input_ts, cmd)

    def test_method(self):
        input_ts = msprime.simulate(10, random_seed=1)
        cmd = "1 --method inside_outside"
        self.verify(input_ts, cmd)
        cmd = "1 --method maximization"
        self.assertRaises(ValueError, self.verify, input_ts, cmd)

    def test_compare_python_api(self):
        input_ts = msprime.simulate(
            100,
            Ne=10000,
            mutation_rate=1e-8,
            recombination_rate=1e-8,
            length=2e4,
            random_seed=10,
        )
        cmd = "10000 -m 1e-8 --method inside_outside"
        self.verify(input_ts, cmd)
        self.compare_python_api(input_ts, cmd, 10000, 1e-8, "inside_outside")
        cmd = "10000 -m 1e-8 --method maximization"
        self.verify(input_ts, cmd)
        self.compare_python_api(input_ts, cmd, 10000, 1e-8, "maximization")

    def preprocess_compare_python_api(self, input_ts):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_filename = pathlib.Path(tmpdir) / "input.trees"
            input_ts.dump(input_filename)
            output_filename = pathlib.Path(tmpdir) / "output.trees"
            full_cmd = "preprocess " + str(input_filename) + f" {output_filename}"
            print(full_cmd)
            cli.tsdate_main(full_cmd.split())
            output_ts = tskit.load(output_filename)
        preprocessed_ts = tsdate.preprocess_ts(input_ts)
        self.ts_equal(output_ts, preprocessed_ts, times_equal=True)

    def test_preprocess_compare_python_api(self):
        input_ts = msprime.simulate(
            100,
            Ne=10000,
            mutation_rate=1e-8,
            recombination_rate=1e-8,
            length=2e4,
            random_seed=10,
        )
        self.preprocess_compare_python_api(input_ts)
