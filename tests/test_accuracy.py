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
import scipy
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
            pytest test_accuracy.py::TestAccuracy::test_make_static_files
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
        "ts_name,min_r2_ts,min_r2_unconstrained,min_spear_ts,min_spear_unconstrained",
        [
            ("one_tree", 0.98601, 0.98601, 0.97719, 0.97719),
            ("few_trees", 0.98220, 0.98220, 0.97744, 0.97744),
            ("many_trees", 0.93449, 0.93449, 0.964547, 0.964547),
        ],
    )
    def test_basic(
        self,
        ts_name,
        min_r2_ts,
        min_r2_unconstrained,
        min_spear_ts,
        min_spear_unconstrained,
        request,
    ):
        ts = tskit.load(
            os.path.join(request.fspath.dirname, "data", ts_name + ".trees")
        )

        sim_ancestry_parameters = json.loads(ts.provenance(0).record)["parameters"]
        assert sim_ancestry_parameters["command"] == "sim_ancestry"
        Ne = sim_ancestry_parameters["population_size"]

        sim_mutations_parameters = json.loads(ts.provenance(1).record)["parameters"]
        assert sim_mutations_parameters["command"] == "sim_mutations"
        mu = sim_mutations_parameters["rate"]

        dts, posteriors = tsdate.inside_outside(
            ts, population_size=Ne, mutation_rate=mu, return_posteriors=True
        )
        # make sure we can read node metadata - old tsdate versions didn't set a schema
        if dts.table_metadata_schemas.node.schema is None:
            tables = dts.dump_tables()
            tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
            dts = tables.tree_sequence()

        # Only test nonsample node times
        nonsamples = np.ones(ts.num_nodes, dtype=bool)
        nonsamples[ts.samples()] = False

        min_vals = {
            "r_sq": {"ts": min_r2_ts, "unconstr": min_r2_unconstrained},
            "spearmans_r": {"ts": min_spear_ts, "unconstr": min_spear_unconstrained},
        }

        expected = ts.nodes_time[nonsamples]
        for observed, src in [
            (dts.nodes_time[nonsamples], "ts"),
            ([dts.node(i).metadata["mn"] for i in np.where(nonsamples)[0]], "unconstr"),
        ]:
            # Test the tree sequence times
            r_sq = np.corrcoef(expected, observed)[0, 1] ** 2
            assert r_sq >= min_vals["r_sq"][src]

            spearmans_r = scipy.stats.spearmanr(expected, observed).correlation
            assert spearmans_r >= min_vals["spearmans_r"][src]

    @pytest.mark.parametrize("Ne", [0.1, 1, 400])
    def test_scaling(self, Ne):
        """
        Test that we are in the right theoretical ballpark given known Ne
        """
        ts = tskit.Tree.generate_comb(2).tree_sequence
        dts = tsdate.inside_outside(ts, population_size=Ne, mutation_rate=None)
        # Check the date is within 10% of the expected
        assert 0.9 < dts.node(dts.first().root).time / (2 * Ne) < 1.1

    @pytest.mark.parametrize(
        "bkwd_rate, trio_tmrca",
        [  # calculated from simulations
            (-1.0, 0.76),
            (-0.9, 0.79),
            (-0.8, 0.82),
            (-0.7, 0.85),
            (-0.6, 0.89),
            (-0.5, 0.94),
            (-0.4, 0.99),
            (-0.3, 1.05),
            (-0.2, 1.12),
            (-0.1, 1.21),
            (0.0, 1.32),
        ],
    )
    def test_piecewise_scaling(self, bkwd_rate, trio_tmrca):
        """
        Test that we are in the right theoretical ballpark given known Ne,
        under exponential growth.

        Check coalescence time of a trio instead of a pair, because of
        https://github.com/tskit-dev/tsdate/issues/230
        """
        time = np.linspace(0, 10, 100)
        ne = 0.5 * np.exp(bkwd_rate * time)
        ts = tskit.Tree.generate_comb(3).tree_sequence
        demo = tsdate.demography.PopulationSizeHistory(ne, time[1:])
        dts = tsdate.inside_outside(ts, population_size=demo, mutation_rate=None)
        # Check the date is within 10% of the expected
        assert 0.9 < dts.node(dts.first().root).time / trio_tmrca < 1.1
