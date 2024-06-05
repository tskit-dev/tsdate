# MIT License
#
# Copyright (c) 2024 Tskit Developers
# Copyright (c) 2020-2024 University of Oxford
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
import logging
from unittest import mock

import msprime
import numpy as np
import pytest
import tskit

import tsdate
import tsdate.cli as cli
from tsdate import __main__ as main

logging_flags = {"-v": "INFO", "--verbosity": "INFO", "-vv": "DEBUG"}


class TestTsdateArgParser:
    """
    Tests for the tsdate argument parser.
    """

    infile = "tmp.trees"
    output = "output.trees"

    def test_default_values(self):
        with mock.patch("tsdate.cli.setup_logging"):
            parser = cli.tsdate_cli_parser()
            args = parser.parse_args(["date", self.infile, self.output])
        assert args.tree_sequence == self.infile
        assert args.output == self.output
        assert args.population_size is None
        assert args.mutation_rate is None
        assert args.recombination_rate is None
        assert args.epsilon == 1e-6
        assert args.num_threads is None
        assert args.probability_space is None  # Use the defaults
        assert args.method == "variational_gamma"
        assert not args.progress

    def test_mutation_rate(self):
        parser = cli.tsdate_cli_parser()
        args = parser.parse_args(["date", self.infile, self.output, "-m", "1e10"])
        assert args.mutation_rate == 1e10
        args = parser.parse_args(
            ["date", self.infile, self.output, "-m", "1e10", "--mutation-rate", "1e10"]
        )
        assert args.mutation_rate == 1e10

    def test_recombination_rate(self):
        parser = cli.tsdate_cli_parser()
        params = ["-m", "1e10"]
        args = parser.parse_args(
            ["date", self.infile, self.output] + params + ["-r", "1e-100"]
        )
        assert args.recombination_rate == 1e-100
        args = parser.parse_args(
            ["date", self.infile, self.output] + params + ["--recombination-rate", "73"]
        )
        assert args.recombination_rate == 73

    def test_epsilon(self):
        parser = cli.tsdate_cli_parser()
        args = parser.parse_args(
            ["date", self.infile, self.output, "-m", "10", "-e", "123"]
        )
        assert args.epsilon == 123
        args = parser.parse_args(
            ["date", self.infile, self.output, "-m", "10", "--epsilon", "321"]
        )
        assert args.epsilon == 321

    def test_num_threads(self):
        parser = cli.tsdate_cli_parser()
        params = ["--method", "maximization", "--num-threads"]
        args = parser.parse_args(["date", self.infile, self.output] + params + ["1"])
        assert args.num_threads == 1
        args = parser.parse_args(["date", self.infile, self.output] + params + ["2"])
        assert args.num_threads == 2

    def test_probability_space(self):
        parser = cli.tsdate_cli_parser()
        params = ["--method", "inside_outside", "--probability-space"]
        args = parser.parse_args(
            ["date", self.infile, self.output] + params + ["linear"]
        )
        assert args.probability_space == "linear"
        args = parser.parse_args(
            ["date", self.infile, self.output] + params + ["logarithmic"]
        )
        assert args.probability_space == "logarithmic"

    @pytest.mark.parametrize("flag, log_status", logging_flags.items())
    def test_verbosity(self, flag, log_status):
        parser = cli.tsdate_cli_parser()
        args = parser.parse_args(["preprocess", self.infile, self.output, flag])
        log_string = cli.setup_logging(args)
        assert log_string == log_status
        assert hasattr(logging, log_string)

    @pytest.mark.parametrize(
        "method", ["inside_outside", "maximization", "variational_gamma"]
    )
    def test_method(self, method):
        parser = cli.tsdate_cli_parser()
        params = ["-m", "1e-8", "--method", method]
        if method != "variational_gamma":
            params += ["-n", "10"]
        args = parser.parse_args(["date", self.infile, self.output] + params)
        assert args.method == method

    def test_progress(self):
        parser = cli.tsdate_cli_parser()
        args = parser.parse_args(
            ["date", self.infile, self.output, "-m", "1e-8", "--progress"]
        )
        assert args.progress

    def test_default_values_preprocess(self):
        with mock.patch("tsdate.cli.setup_logging"):
            parser = cli.tsdate_cli_parser()
            args = parser.parse_args(["preprocess", self.infile, self.output])
        assert args.tree_sequence == self.infile
        assert args.output == self.output
        assert args.minimum_gap == 1000000
        assert args.trim_telomeres
        assert args.split_disjoint


class TestMain:
    def test_main(self, capfd):
        # Just for coverage really
        with pytest.raises(SystemExit):
            main.main()
        captured = capfd.readouterr()
        assert "subcommand" in captured.err


class RunCLI:
    mutation_rate = 1
    popsize = 1

    def run_tsdate_cli(self, tmp_path, input_ts, params=None, cmd="date"):
        if cmd == "date" and params is None:
            params = str(self.popsize)
        input_filename = tmp_path / "input.trees"
        input_ts.dump(input_filename)
        output_filename = tmp_path / "output.trees"
        cmd = [cmd, str(input_filename), str(output_filename)]
        if params is not None:
            cmd.append(params)
        cli.tsdate_main(" ".join(cmd).split())
        return tskit.load(output_filename)


class TestCLIErrors(RunCLI):
    @pytest.mark.parametrize("cmd", ["date", "preprocess"])
    def test_bad_file(self, tmp_path, cmd):
        input_filename = tmp_path / "input.trees"
        print("bad file", file=input_filename.open("w"))
        output_filename = tmp_path / "output.trees"
        cmds = [cmd, str(input_filename), str(output_filename)]
        if cmd == "date":
            cmds.append("-m")
            cmds.append(str(self.mutation_rate))
        with pytest.raises(SystemExit, match="FileFormatError"):
            cli.tsdate_main(cmds)

    def test_bad_date_method(self, tmp_path, capfd):
        bad = "bad_method"
        input_ts = msprime.simulate(4, random_seed=123)
        params = f"--method {bad}"
        with pytest.raises(SystemExit):
            self.run_tsdate_cli(tmp_path, input_ts, params)
        captured = capfd.readouterr()
        assert bad in captured.err

    def test_bad_rescaling_io(self, tmp_path):
        input_ts = msprime.simulate(4, random_seed=123)
        params = f"-n {self.popsize} --method inside_outside"
        with pytest.raises(SystemExit, match="not currently used"):
            self.run_tsdate_cli(tmp_path, input_ts, params + " --rescaling-intervals 1")

    def test_bad_max_it_io(self, tmp_path):
        input_ts = msprime.simulate(4, random_seed=123)
        params = f"-n {self.popsize} --method inside_outside"
        with pytest.raises(SystemExit, match="not currently used"):
            self.run_tsdate_cli(tmp_path, input_ts, params + " --max-iterations 5")


class TestOutput(RunCLI):
    """
    Tests for the command-line output.
    """

    def test_no_output_inside_outside(self, tmp_path, capfd):
        ts = msprime.simulate(4, random_seed=123)
        self.run_tsdate_cli(tmp_path, ts, f"-n {self.popsize} --method inside_outside")
        (out, err) = capfd.readouterr()
        assert out == ""
        assert err == ""

    def test_no_output_variational_gamma(self, tmp_path, capfd):
        ts = msprime.simulate(4, mutation_rate=10, random_seed=123)
        self.run_tsdate_cli(tmp_path, ts, "-m 10")
        (out, err) = capfd.readouterr()
        assert out == ""
        assert err == ""

    @pytest.mark.parametrize("flag, log_status", logging_flags.items())
    def test_verbosity(self, tmp_path, caplog, flag, log_status):
        popsize = 10000
        ts = msprime.simulate(
            10,
            Ne=popsize,
            mutation_rate=1e-8,
            recombination_rate=1e-8,
            length=2e4,
            random_seed=10,
        )
        # Have to set the log level on the caplog object, because
        # logging.basicConfig() doesn't work within pytest
        caplog.set_level(getattr(logging, log_status))
        # either tsdate preprocess or tsdate date (in_out method has debug asserts)
        self.run_tsdate_cli(tmp_path, ts, flag, cmd="preprocess")
        self.run_tsdate_cli(tmp_path, ts, f"-n 10 --method inside_outside {flag}")
        assert log_status in caplog.text

    @pytest.mark.parametrize(
        "method", ["inside_outside", "maximization", "variational_gamma"]
    )
    def test_no_progress(self, method, tmp_path, capfd):
        input_ts = msprime.simulate(4, mutation_rate=10, random_seed=123)
        params = f"-m 10 --method {method}"
        if method != "variational_gamma":
            params += f" -n {self.popsize}"
        self.run_tsdate_cli(tmp_path, input_ts, params)
        (out, err) = capfd.readouterr()
        assert out == ""
        # run_tsdate_cli print logging to stderr
        assert err == ""

    def test_progress(self, tmp_path, capfd):
        input_ts = msprime.simulate(4, random_seed=123)
        params = "--method inside_outside -n 10 --progress"
        self.run_tsdate_cli(tmp_path, input_ts, params)
        (out, err) = capfd.readouterr()
        assert out == ""
        # run_tsdate_cli print logging to stderr
        desc = (
            "Find Node Spans",
            "TipCount",
            "Find Mixture Priors",
            "Inside",
            "Outside",
        )
        for match in desc:
            assert match in err
        assert err.count("100%") == len(desc)
        assert err.count("it/s") >= len(desc)

    def test_iterative_progress(self, tmp_path, capfd):
        input_ts = msprime.simulate(4, mutation_rate=10, random_seed=123)
        params = "--method variational_gamma --mutation-rate 10 "
        params += "--progress --rescaling-intervals 0"
        self.run_tsdate_cli(tmp_path, input_ts, params)
        (out, err) = capfd.readouterr()
        assert out == ""
        # run_tsdate_cli print logging to stderr
        assert err.count("Expectation Propagation: 100%") == 1
        # The capfd fixture doesn't end up capturing progress bars with
        # leave=False (they get deleted) so we can't see these in the output
        # assert err.count("Iteration 1: 100%") == 1
        # assert err.count("Rootwards: 100%") > 1
        # assert err.count("Rootwards: 100%") == err.count("Leafwards: 100%")


class TestEndToEnd(RunCLI):
    """
    Class to test input to CLI outputs dated tree sequences.
    """

    def ts_equal(self, ts1, ts2, times_equal=False):
        assert ts1.sequence_length == ts2.sequence_length
        t1 = ts1.tables
        t2 = ts2.tables
        assert t1.sites == t2.sites
        # Edges may have been re-ordered, since sortedness requirements specify
        # they are sorted by parent time, and the relative order of
        # (unconnected) parent nodes might have changed due to time inference
        assert set(t1.edges) == set(t2.edges)
        if not times_equal:
            # The dated and undated tree sequences should not have the same node times
            assert not np.array_equal(ts1.tables.nodes.time, ts2.tables.nodes.time)
            # New tree sequence will have node times in metadata and will no longer have
            # mutation times
            for column_name in t1.nodes.column_names:
                if column_name not in ["time", "metadata", "metadata_offset"]:
                    col_t1 = getattr(t1.nodes, column_name)
                    col_t2 = getattr(t2.nodes, column_name)
                    assert np.array_equal(col_t1, col_t2)
            for column_name in t1.mutations.column_names:
                if column_name not in ["time", "metadata", "metadata_offset"]:
                    col_t1 = getattr(t1.mutations, column_name)
                    col_t2 = getattr(t2.mutations, column_name)
                    assert np.array_equal(col_t1, col_t2)
            # Assert that last provenance shows tree sequence was dated
            assert len(t1.provenances) == len(t2.provenances) - 1
            for index, (prov1, prov2) in enumerate(zip(t1.provenances, t2.provenances)):
                assert prov1 == prov2
                if index == len(t1.provenances) - 1:
                    break
            assert json.loads(t2.provenances[-1].record)["software"]["name"] == "tsdate"

        else:
            assert t1.nodes == t2.nodes

    def verify(self, tmp_path, input_ts, params=None):
        output_ts = self.run_tsdate_cli(tmp_path, input_ts, params)
        assert input_ts.num_samples == output_ts.num_samples
        self.ts_equal(input_ts, output_ts)

    def compare_python_api(self, tmp_path, input_ts, params, Ne, mutation_rate, method):
        output_ts = self.run_tsdate_cli(tmp_path, input_ts, params)
        popsize = None if method == "variational_gamma" else Ne
        dated_ts = tsdate.date(
            input_ts,
            population_size=popsize,
            mutation_rate=mutation_rate,
            method=method,
        )
        assert np.array_equal(dated_ts.nodes_time, output_ts.nodes_time)

    def test_ts(self, tmp_path):
        input_ts = msprime.simulate(10, mutation_rate=4, random_seed=1)
        params = "--mutation-rate 4"
        self.verify(tmp_path, input_ts, params)

    def test_mutation_rate(self, tmp_path):
        input_ts = msprime.simulate(10, mutation_rate=4, random_seed=1)
        params = "--mutation-rate 4"
        self.verify(tmp_path, input_ts, params)

    def test_recombination_rate(self, tmp_path):
        input_ts = msprime.simulate(10, mutation_rate=4, random_seed=1)
        params = "-m 4 --recombination-rate 1e-8"
        with pytest.raises(NotImplementedError):
            self.verify(tmp_path, input_ts, params)
        params = "--method inside_outside -n 1 --recombination-rate 1e-8"
        with pytest.raises(NotImplementedError):
            self.verify(tmp_path, input_ts, params)

    def test_epsilon(self, tmp_path):
        input_ts = msprime.simulate(10, mutation_rate=4, random_seed=1)
        params = "--mutation-rate 4 --epsilon 1e-3"
        self.verify(tmp_path, input_ts, params)

    def test_num_threads(self, tmp_path):
        input_ts = msprime.simulate(10, random_seed=1)
        params = f"-n {self.popsize} --num-threads 2 --method inside_outside"
        self.verify(tmp_path, input_ts, params)

    def test_probability_space(self, tmp_path):
        input_ts = msprime.simulate(10, random_seed=1)
        params = f"-n {self.popsize} --probability-space linear --method inside_outside"
        self.verify(tmp_path, input_ts, params)
        params = (
            f"-n {self.popsize} --probability-space logarithmic "
            "--method inside_outside"
        )
        self.verify(tmp_path, input_ts, params)

    def test_method(self, tmp_path):
        input_ts = msprime.simulate(10, random_seed=1)
        params = f"-n {self.popsize} --method inside_outside"
        self.verify(tmp_path, input_ts, params)
        params = f"-n {self.popsize} --method maximization"
        with pytest.raises(ValueError):
            self.verify(tmp_path, input_ts, params)

    @pytest.mark.parametrize(
        "method", ["inside_outside", "maximization", "variational_gamma"]
    )
    def test_compare_python_api(self, tmp_path, method):
        popsize = 10000
        input_ts = msprime.simulate(
            100,
            Ne=popsize,
            mutation_rate=1e-8,
            recombination_rate=1e-8,
            length=2e4,
            random_seed=10,
        )
        params = f"-m 1e-8 --method {method}"
        if method != "variational_gamma":
            params += f" -n {popsize}"
        self.verify(tmp_path, input_ts, params)
        self.compare_python_api(tmp_path, input_ts, params, popsize, 1e-8, method)

    def preprocess_compare_python_api(self, tmp_path, input_ts):
        output_ts = self.run_tsdate_cli(tmp_path, input_ts, cmd="preprocess")
        preprocessed_ts = tsdate.preprocess_ts(input_ts)
        self.ts_equal(output_ts, preprocessed_ts, times_equal=True)

    def test_preprocess_compare_python_api(self, tmp_path):
        input_ts = msprime.simulate(
            100,
            Ne=10000,
            mutation_rate=1e-8,
            recombination_rate=1e-8,
            length=2e4,
            random_seed=10,
        )
        self.preprocess_compare_python_api(tmp_path, input_ts)
