# MIT License
#
# Copyright (c) 2021-24 Tskit Developers
# Copyright (c) 2020 University of Oxford
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Test cases for the python API for tsdate.
"""
import logging

import msprime
import numpy as np
import pytest
import tsinfer
import tskit
import utility_functions

import tsdate
from tsdate.base import LIN
from tsdate.base import LOG
from tsdate.demography import PopulationSizeHistory
from tsdate.evaluation import remove_edges
from tsdate.evaluation import unsupported_edges


class TestConstants:
    def test_matches_tsinfer_consts(self):
        assert tsinfer.NODE_IS_HISTORICAL_SAMPLE == tsdate.NODE_IS_HISTORICAL_SAMPLE


class TestPrebuilt:
    """
    Tests for tsdate on prebuilt tree sequences
    """

    def test_invalid_method_failure(self):
        ts = utility_functions.two_tree_mutation_ts()
        with pytest.raises(ValueError, match="method must be one of"):
            tsdate.date(ts, population_size=1, mutation_rate=None, method="foo")

    def test_no_mutations_failure(self):
        ts = utility_functions.single_tree_ts_n2()
        with pytest.raises(ValueError, match="No mutations present"):
            tsdate.variational_gamma(ts, mutation_rate=1)

    def test_no_population_size(self):
        ts = utility_functions.two_tree_mutation_ts()
        with pytest.raises(ValueError, match="Must specify population size"):
            tsdate.inside_outside(ts, mutation_rate=None)

    def test_no_mutation(self):
        ts = utility_functions.two_tree_mutation_ts()
        with pytest.raises(ValueError, match="method requires mutation rate"):
            tsdate.date(
                ts, method="maximization", population_size=1, mutation_rate=None
            )
        with pytest.raises(ValueError, match="method requires mutation rate"):
            tsdate.date(ts, method="variational_gamma", mutation_rate=None)

    def test_not_needed_population_size(self):
        ts = utility_functions.two_tree_mutation_ts()
        prior = tsdate.build_prior_grid(ts, population_size=1, timepoints=10)
        with pytest.raises(ValueError, match="Cannot specify population size"):
            tsdate.inside_outside(
                ts, population_size=1, mutation_rate=None, priors=prior
            )

    def test_bad_population_size(self):
        ts = utility_functions.two_tree_mutation_ts()
        for Ne in [0, -1]:
            with pytest.raises(ValueError, match="greater than 0"):
                tsdate.inside_outside(ts, mutation_rate=None, population_size=Ne)

    def test_both_ne_and_population_size_specified(self):
        ts = utility_functions.two_tree_mutation_ts()
        with pytest.raises(ValueError, match="Only provide one of Ne"):
            tsdate.inside_outside(
                ts, mutation_rate=None, population_size=PopulationSizeHistory(1), Ne=1
            )
        tsdate.inside_outside(ts, mutation_rate=None, Ne=PopulationSizeHistory(1))

    def test_inside_outside_dangling_failure(self):
        ts = utility_functions.single_tree_ts_n2_dangling()
        with pytest.raises(ValueError, match="simplified"):
            tsdate.inside_outside(ts, mutation_rate=None, population_size=1)

    def test_variational_gamma_dangling(self):
        # Dangling nodes are fine for the variational gamma method
        ts = utility_functions.single_tree_ts_n2_dangling()
        ts = msprime.sim_mutations(ts, rate=2, random_seed=1)
        assert ts.num_mutations > 1
        tsdate.variational_gamma(ts, mutation_rate=2)

    def test_inside_outside_unary_failure(self):
        ts = utility_functions.single_tree_ts_with_unary()
        with pytest.raises(ValueError, match="unary"):
            tsdate.inside_outside(ts, mutation_rate=None, population_size=1)

    @pytest.mark.skip("V_gamma should fail with unary nodes, but doesn't currently")
    def test_variational_gamma_unary_failure(self):
        ts = utility_functions.single_tree_ts_with_unary()
        ts = msprime.sim_mutations(ts, rate=1, random_seed=1)
        with pytest.raises(ValueError, match="unary"):
            tsdate.variational_gamma(ts, mutation_rate=1)

    @pytest.mark.parametrize("probability_space", (LOG, LIN))
    @pytest.mark.parametrize("mu", (None, 1))
    def test_fails_with_recombination(self, probability_space, mu):
        ts = utility_functions.two_tree_mutation_ts()
        with pytest.raises(NotImplementedError):
            tsdate.inside_outside(
                ts,
                mutation_rate=mu,
                population_size=1,
                recombination_rate=1,
                probability_space=probability_space,
            )

    def test_default_time_units(self):
        ts = utility_functions.two_tree_mutation_ts()
        ts = tsdate.date(ts, mutation_rate=1)
        assert ts.time_units == "generations"

    def test_default_alternative_time_units(self):
        ts = utility_functions.two_tree_mutation_ts()
        ts = tsdate.date(ts, mutation_rate=1, time_units="years")
        assert ts.time_units == "years"

    def test_no_posteriors(self):
        ts = utility_functions.two_tree_mutation_ts()
        with pytest.raises(ValueError, match="Cannot return posterior"):
            tsdate.date(
                ts,
                population_size=1,
                return_posteriors=True,
                method="maximization",
                mutation_rate=1,
            )

    def test_discretised_posteriors(self):
        ts = utility_functions.two_tree_mutation_ts()
        ts, posteriors = tsdate.inside_outside(
            ts, mutation_rate=None, population_size=1, return_posteriors=True
        )
        assert len(posteriors) == ts.num_nodes - ts.num_samples + 1
        assert len(posteriors["time"]) > 0
        for node in ts.nodes():
            if not node.is_sample():
                assert node.id in posteriors
                assert len(posteriors[node.id]) == len(posteriors["time"])
                assert np.isclose(np.sum(posteriors[node.id]), 1)

    def test_variational_posteriors(self):
        """
        There are no time-gridded posteriors returned by variational gamma,
        Output is currently None, but see https://github.com/tskit-dev/tsdate/issues/388
        """
        ts = utility_functions.two_tree_mutation_ts()
        ts, posteriors = tsdate.date(
            ts,
            mutation_rate=1e-2,
            method="variational_gamma",
            return_posteriors=True,
        )
        assert posteriors is None

    def test_marginal_likelihood(self):
        ts = utility_functions.two_tree_mutation_ts()
        _, _, marg_lik = tsdate.inside_outside(
            ts,
            mutation_rate=None,
            population_size=1,
            return_posteriors=True,
            return_likelihood=True,
        )
        _, marg_lik_again = tsdate.inside_outside(
            ts, mutation_rate=None, population_size=1, return_likelihood=True
        )
        assert marg_lik == marg_lik_again

    def test_intervals(self):
        ts = utility_functions.two_tree_ts()
        long_ts = utility_functions.two_tree_ts_extra_length()
        keep_ts = long_ts.keep_intervals([[0.0, 1.0]])
        delete_ts = long_ts.delete_intervals([[1.0, 1.5]])
        dated_ts = tsdate.inside_outside(ts, mutation_rate=None, population_size=1)
        dated_keep_ts = tsdate.inside_outside(
            keep_ts, mutation_rate=None, population_size=1
        )
        dated_deleted_ts = tsdate.inside_outside(
            delete_ts, mutation_rate=None, population_size=1
        )
        assert np.allclose(
            dated_ts.tables.nodes.time[:], dated_keep_ts.tables.nodes.time[:]
        )
        assert np.allclose(
            dated_ts.tables.nodes.time[:], dated_deleted_ts.tables.nodes.time[:]
        )


class TestSimulated:
    """
    Tests for tsdate on simulated tree sequences.
    """

    def ts_equal_except_times(self, ts1, ts2):
        assert ts1.sequence_length == ts2.sequence_length
        t1 = ts1.tables
        t2 = ts2.tables
        assert t1.sites == t2.sites
        # Edges may have been re-ordered, since sortedness requirements specify
        # they are sorted by parent time, and the relative order of
        # (unconnected) parent nodes might have changed due to time inference
        assert set(t1.edges) == set(t2.edges)
        # The dated and undated tree sequences should not have the same node times
        assert not np.array_equal(ts1.tables.nodes.time, ts2.tables.nodes.time)
        # New tree sequence will have node times in metadata and all mutation times
        # set to tskit.UNKNOWN_TIME
        for column_name in t1.nodes.column_names:
            if column_name not in ["time", "metadata", "metadata_offset"]:
                col_t1 = t1.nodes.column_names
                col_t2 = t2.nodes.column_names
                assert np.array_equal(col_t1, col_t2)
        for column_name in t1.mutations.column_names:
            if column_name not in ["time"]:
                col_t1 = t1.mutations.column_names
                col_t2 = t2.mutations.column_names
                assert np.array_equal(col_t1, col_t2)
        # Assert that last provenance shows tree sequence was dated
        assert len(t1.provenances) == len(t2.provenances) - 1
        for index, (prov1, prov2) in enumerate(zip(t1.provenances, t2.provenances)):
            assert prov1 == prov2
            if index == len(t1.provenances) - 1:
                break

    @pytest.mark.parametrize("rho", [0, 5])
    @pytest.mark.parametrize("method", tsdate.estimation_methods.keys())
    def test_simple_sim(self, rho, method):
        ts = msprime.simulate(8, mutation_rate=5, recombination_rate=rho, random_seed=2)
        if rho != 0:
            assert ts.num_trees > 1
        Ne = None if method == "variational_gamma" else 1
        dated_ts = tsdate.date(ts, population_size=Ne, mutation_rate=5, method=method)
        self.ts_equal_except_times(ts, dated_ts)

    def test_simple_sim_larger_example(self):
        # This makes ~1700 trees, and previously caused a failure
        ts = msprime.simulate(
            sample_size=10,
            length=2e6,
            Ne=10000,
            mutation_rate=1e-8,
            recombination_rate=1e-8,
            random_seed=11,
        )
        io_ts = tsdate.inside_outside(ts, population_size=10000, mutation_rate=1e-8)
        max_ts = tsdate.maximization(ts, population_size=10000, mutation_rate=1e-8)
        self.ts_equal_except_times(ts, io_ts)
        self.ts_equal_except_times(ts, max_ts)

    def test_linear_space(self):
        # This makes ~1700 trees, and previously caused a failure
        ts = msprime.simulate(
            sample_size=10,
            length=2e6,
            Ne=10000,
            mutation_rate=1e-8,
            recombination_rate=1e-8,
            random_seed=11,
        )
        priors = tsdate.build_prior_grid(
            ts, population_size=10000, timepoints=10, approximate_priors=None
        )
        dated_ts = tsdate.inside_outside(
            ts, mutation_rate=1e-8, priors=priors, probability_space=LIN
        )
        maximized_ts = tsdate.maximization(
            ts,
            mutation_rate=1e-8,
            priors=priors,
            probability_space=LIN,
        )
        self.ts_equal_except_times(ts, dated_ts)
        self.ts_equal_except_times(ts, maximized_ts)

    def test_with_unary(self):
        ts = msprime.simulate(
            8,
            mutation_rate=10,
            recombination_rate=10,
            record_full_arg=True,
            random_seed=12,
        )

        with pytest.raises(ValueError, match="unary"):
            tsdate.maximization(ts, population_size=1, mutation_rate=10)
        with pytest.raises(ValueError, match="unary"):
            tsdate.inside_outside(ts, population_size=1, mutation_rate=10)

    def test_fails_multi_root(self):
        ts = msprime.simulate(8, mutation_rate=2, random_seed=2)
        tree = ts.first()
        tables = ts.dump_tables()
        tables.edges.clear()
        internal_edge_removed = False
        for row in ts.tables.edges:
            if row.parent not in tree.roots and row.child not in ts.samples():
                if not internal_edge_removed:
                    continue
            tables.edges.append(row)
        multiroot_ts = tables.tree_sequence()
        good_priors = tsdate.build_prior_grid(ts, population_size=1)
        with pytest.raises(ValueError):
            tsdate.build_prior_grid(multiroot_ts, population_size=1)
        with pytest.raises(ValueError):
            tsdate.date(multiroot_ts, population_size=1, mutation_rate=2)
        with pytest.raises(ValueError):
            tsdate.date(
                multiroot_ts, population_size=1, mutation_rate=2, priors=good_priors
            )

    def test_non_contemporaneous(self):
        samples = [
            msprime.Sample(population=0, time=0),
            msprime.Sample(population=0, time=0),
            msprime.Sample(population=0, time=0),
            msprime.Sample(population=0, time=1.0),
        ]
        ts = msprime.simulate(samples=samples, Ne=1, mutation_rate=2, random_seed=12)
        with pytest.raises(ValueError, match="noncontemporaneous"):
            tsdate.inside_outside(ts, population_size=1, mutation_rate=2)

    def test_mutation_times(self):
        ts = msprime.simulate(20, Ne=1, mutation_rate=1, random_seed=12)
        assert np.all(ts.tables.mutations.time > 0)
        dated = tsdate.date(ts, mutation_rate=1)
        assert np.all(~np.isnan(dated.tables.mutations.time))

    @pytest.mark.skip("YAN to fix")
    def test_truncated_ts(self):
        Ne = 1e2
        mu = 2e-4
        ts = msprime.simulate(
            10,
            Ne=Ne,
            length=400,
            recombination_rate=1e-4,
            mutation_rate=mu,
            random_seed=12,
        )
        truncated_ts = utility_functions.truncate_ts_samples(
            ts, average_span=200, random_seed=123
        )
        dated_ts = tsdate.date(truncated_ts, population_size=Ne, mutation_rate=mu)
        # We should ideally test whether *haplotypes* are the same here
        # in case allele encoding has changed. But haplotypes() doesn't currently
        # deal with missing data
        self.ts_equal_except_times(truncated_ts, dated_ts)


class TestInferred:
    """
    Tests for tsdate on simulated then inferred tree sequences.
    """

    @pytest.mark.parametrize("rho", [0, 5])
    @pytest.mark.parametrize("method", tsdate.estimation_methods.keys())
    def test_simple_sim(self, rho, method):
        ts = msprime.simulate(8, mutation_rate=5, recombination_rate=rho, random_seed=2)
        if rho != 0:
            assert ts.num_trees > 1
        for u_tm in [True, False]:
            sample_data = tsinfer.SampleData.from_tree_sequence(ts, use_sites_time=u_tm)
            i_ts = tsinfer.infer(sample_data).simplify()
            Ne = None if method == "variational_gamma" else 1
            d_ts = tsdate.date(i_ts, population_size=Ne, mutation_rate=5, method=method)
            assert all([a == b for a, b in zip(ts.haplotypes(), d_ts.haplotypes())])


class TestVariational:
    """
    Tests for tsdate with variational algorithm
    """

    @pytest.fixture(autouse=True)
    def ts(self):
        ts = msprime.sim_ancestry(
            samples=10,
            recombination_rate=1e-8,
            sequence_length=1e5,
            population_size=1e4,
            random_seed=2,
        )
        self.ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=1)

    def test_binary(self):
        tsdate.variational_gamma(self.ts, mutation_rate=1e-8)

    def test_polytomy(self):
        pts = remove_edges(self.ts, unsupported_edges(self.ts)).simplify()
        tsdate.variational_gamma(pts, mutation_rate=1e-8)

    def test_inferred(self):
        its = tsinfer.infer(tsinfer.SampleData.from_tree_sequence(self.ts)).simplify()
        tsdate.variational_gamma(its, mutation_rate=1e-8)

    def test_bad_arguments(self):
        with pytest.raises(ValueError, match="Maximum number of EP iterations"):
            tsdate.variational_gamma(self.ts, mutation_rate=5, max_iterations=-1)

    def test_no_existing_mutation_metadata(self):
        # Currently only the variational_gamma method embeds mutation metadata
        ts = tsdate.variational_gamma(self.ts, mutation_rate=1e-8)
        for m in ts.mutations():
            assert "mn" in m.metadata
            assert "vr" in m.metadata
            assert m.metadata["mn"] > 0
            assert m.metadata["vr"] > 0

    def test_existing_mutation_metadata(self, caplog):
        tables = self.ts.dump_tables()
        m = self.ts.mutation(-1)
        tables.mutations.truncate(self.ts.num_mutations - 1)
        tables.mutations.metadata_schema = tskit.MetadataSchema.permissive_json()
        tables.mutations.append(m.replace(metadata={"test": 7}))
        ts = tables.tree_sequence()
        with caplog.at_level(logging.WARNING):
            dts = tsdate.variational_gamma(ts, mutation_rate=1e-8)
            assert caplog.text == ""
        assert dts.mutation(-1).metadata["test"] == 7
        for m in dts.mutations():
            assert m.metadata["mn"] > 0
            assert m.metadata["vr"] > 0

    def test_existing_byte_mutation_metadata(self, caplog):
        tables = self.ts.dump_tables()
        m = self.ts.mutation(-1)
        tables.mutations.truncate(self.ts.num_mutations - 1)
        tables.mutations.append(m.replace(metadata=b"x"))
        ts = tables.tree_sequence()
        with caplog.at_level(logging.WARNING):
            dts = tsdate.variational_gamma(ts, mutation_rate=1e-8)
            assert "Could not set" in caplog.text
        assert dts.mutation(-1).metadata == b"x"

    def test_existing_struct_mutation_metadata(self, caplog):
        tables = self.ts.dump_tables()
        tables.mutations.metadata_schema = tskit.MetadataSchema(
            {
                "codec": "struct",
                "type": "object",
                "properties": {
                    "mn": {
                        "type": "number",
                        "binaryFormat": "f",
                        "description": "Posterior mean mutation time",
                    },
                    "vr": {
                        "type": "number",
                        "binaryFormat": "f",
                        "description": "Posterior variance in mutation time",
                    },
                },
                "additionalProperties": False,
            }
        )
        tables.mutations.packset_metadata(
            [
                tables.mutations.metadata_schema.validate_and_encode_row(
                    {"mn": 0, "vr": 0}
                )
                for _ in range(tables.mutations.num_rows)
            ]
        )
        ts = tables.tree_sequence()
        with caplog.at_level(logging.WARNING):
            dts = tsdate.variational_gamma(ts, mutation_rate=1e-8)
            assert caplog.text == ""
        for m in dts.mutations():
            assert m.metadata["mn"] > 0
            assert m.metadata["vr"] > 0

    def test_incompatible_schema_mutation_metadata(self, caplog):
        tables = self.ts.dump_tables()
        tables.mutations.metadata_schema = tskit.MetadataSchema(
            {
                "codec": "struct",
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            }
        )
        ts = tables.tree_sequence()
        with caplog.at_level(logging.WARNING):
            dts = tsdate.variational_gamma(ts, mutation_rate=1e-8)
            assert "Could not set" in caplog.text
        assert len(dts.tables.mutations.metadata) == 0
