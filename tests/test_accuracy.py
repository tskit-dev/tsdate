# MIT License
#
# Copyright (c) 2022 University of Oxford
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all
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
Test cases for tsdate accuracy.
"""
import json
import os

import msprime
import numpy as np
import pytest
import tskit

import tsdate


class TestAccuracy:
    """
    Test for some of the basic functions used in tsdate
    """

    @pytest.mark.makefiles
    def test_make_static_files(self, request):
        """
        The function used to create the tree sequences for accuracy testing.
        So that we are assured of using the same tree sequence, regardless of the
        version and random number generator used in msprime, we keep these
        as static files and only run this function when explicitly specified, e.g. via
            pytest test_accuracy.py::TestAccuracy::create_static_files
        """
        mu = 1e-6
        Ne = 1e4
        seed = 123
        for name, rho in zip(
            ["one_tree", "few_trees", "many_trees"],
            [0, 7e-9, 1.3e-7],  # Chosen to give 1, 2, and 25 trees
        ):
            ts = msprime.sim_ancestry(
                10,
                population_size=Ne,
                sequence_length=1e3,
                recombination_rate=rho,
                random_seed=seed,
            )
            if name != "one_tree":
                assert ts.num_trees > 1
            if name == "few_trees":
                assert ts.num_trees < 5
            if name == "many_trees":
                assert ts.num_trees >= 20

            ts = msprime.sim_mutations(ts, rate=mu, random_seed=seed)
            assert ts.num_mutations > 100
            ts.dump(os.path.join(request.fspath.dirname, "data", f"{name}.trees"))

    @pytest.mark.parametrize(
        "ts_name,min_r2_ts,min_r2_posterior",
        [
            ("one_tree", 0.94776615238, 0.94776615238),
            ("few_trees", 0.96605244, 0.96605244),
            ("many_trees", 0.92646, 0.92646),
        ],
    )
    def test_basic(self, ts_name, min_r2_ts, min_r2_posterior, request):
        ts = tskit.load(
            os.path.join(request.fspath.dirname, "data", ts_name + ".trees")
        )

        sim_ancestry_parameters = json.loads(ts.provenance(0).record)["parameters"]
        assert sim_ancestry_parameters["command"] == "sim_ancestry"
        Ne = sim_ancestry_parameters["population_size"]

        sim_mutations_parameters = json.loads(ts.provenance(1).record)["parameters"]
        assert sim_mutations_parameters["command"] == "sim_mutations"
        mu = sim_mutations_parameters["rate"]

        dts, posteriors = tsdate.date(
            ts, Ne=Ne, mutation_rate=mu, return_posteriors=True
        )
        # Only test nonsample node times
        nonsample_nodes = np.ones(ts.num_nodes, dtype=bool)
        nonsample_nodes[ts.samples()] = False

        # Test the tree sequence times
        r_sq = (
            np.corrcoef(
                np.log(ts.nodes_time[nonsample_nodes]),
                np.log(dts.nodes_time[nonsample_nodes]),
            )[0, 1]
            ** 2
        )
        assert r_sq >= min_r2_ts

        # Test the posterior means too.
        post_mean = np.array(
            [
                np.sum(posteriors[i] * posteriors["start_time"]) / np.sum(posteriors[i])
                for i in np.where(nonsample_nodes)[0]
            ]
        )
        r_sq = (
            np.corrcoef(np.log(ts.nodes_time[nonsample_nodes]), np.log(post_mean))[0, 1]
            ** 2
        )
        assert r_sq >= min_r2_posterior
