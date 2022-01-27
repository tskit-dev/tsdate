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
Test cases for the python API for tsdate.
"""
import collections
import json
import unittest
import warnings

import msprime
import numpy as np
import pytest
import scipy
import tsinfer
import tskit
import utility_functions

import tsdate
from tsdate.base import NodeGridValues
from tsdate.core import constrain_ages_topo
from tsdate.core import date
from tsdate.core import get_dates
from tsdate.core import InOutAlgorithms
from tsdate.core import Likelihoods
from tsdate.core import LogLikelihoods
from tsdate.core import posterior_mean_var
from tsdate.prior import ConditionalCoalescentTimes
from tsdate.prior import fill_priors
from tsdate.prior import gamma_approx
from tsdate.prior import PriorParams
from tsdate.prior import SpansBySamples
from tsdate.util import nodes_time_unconstrained


class TestBasicFunctions:
    """
    Test for some of the basic functions used in tsdate
    """

    def test_alpha_prob(self):
        assert ConditionalCoalescentTimes.m_prob(2, 2, 3) == 1.0
        assert ConditionalCoalescentTimes.m_prob(2, 2, 4) == 0.25

    def test_tau_expect(self):
        assert ConditionalCoalescentTimes.tau_expect(10, 10) == 1.8
        assert ConditionalCoalescentTimes.tau_expect(10, 100) == 0.09
        assert ConditionalCoalescentTimes.tau_expect(100, 100) == 1.98
        assert ConditionalCoalescentTimes.tau_expect(5, 10) == 0.4

    def test_tau_squared_conditional(self):
        assert pytest.approx(
            ConditionalCoalescentTimes.tau_squared_conditional(1, 10), 4.3981418
        )
        assert pytest.approx(
            ConditionalCoalescentTimes.tau_squared_conditional(100, 100), 4.87890977e-18
        )

    def test_tau_var(self):
        assert ConditionalCoalescentTimes.tau_var(2, 2) == 1
        assert pytest.approx(ConditionalCoalescentTimes.tau_var(10, 20), 0.0922995960)
        assert pytest.approx(ConditionalCoalescentTimes.tau_var(50, 50), 1.15946186)

    def test_gamma_approx(self):
        assert gamma_approx(2, 1) == (4.0, 2.0)
        assert gamma_approx(0.5, 0.1) == (2.5, 5.0)


class TestNodeTipWeights(unittest.TestCase):
    def verify_weights(self, ts):
        span_data = SpansBySamples(ts)
        # Check all non-sample nodes in a tree are represented
        nonsample_nodes = collections.defaultdict(float)
        for tree in ts.trees():
            for n in tree.nodes():
                if not tree.is_sample(n):
                    # do not count a span of a node where there are no sample descendants
                    nonsample_nodes[n] += tree.span if tree.num_samples(n) > 0 else 0
        assert set(span_data.nodes_to_date) == set(nonsample_nodes.keys())
        for id_, span in nonsample_nodes.items():
            assert pytest.approx(span, span_data.node_spans[id_])
        for focal_node in span_data.nodes_to_date:
            wt = 0
            for _, weights in span_data.get_weights(focal_node).items():
                assert 0 <= focal_node < ts.num_nodes
                wt += np.sum(weights["weight"])
                assert max(weights["descendant_tips"]) <= ts.num_samples
            if not np.isnan(wt):
                # Dangling nodes will have wt=nan
                assert pytest.approx(wt, 1.0)
        return span_data

    def test_one_tree_n2(self):
        ts = utility_functions.single_tree_ts_n2()
        span_data = self.verify_weights(ts)
        # with a single tree there should only be one weight
        for node in span_data.nodes_to_date:
            assert len(span_data.get_weights(node)), 1
        assert 2 in span_data.get_weights(2)[ts.num_samples]["descendant_tips"]

    def test_one_tree_n3(self):
        ts = utility_functions.single_tree_ts_n3()
        n = ts.num_samples
        span_data = self.verify_weights(ts)
        # with a single tree there should only be one weight
        for node in span_data.nodes_to_date:
            assert len(span_data.get_weights(node)), 1
        for nd, expd_tips in [
            (4, 3),  # Node 4 (root) expected to have 3 descendant tips
            (3, 2),
        ]:  # Node 3 (1st internal node) expected to have 2 descendant tips
            assert np.isin(span_data.get_weights(nd)[n]["descendant_tips"], expd_tips)

    def test_one_tree_n4(self):
        ts = utility_functions.single_tree_ts_n4()
        n = ts.num_samples
        span_data = self.verify_weights(ts)
        # with a single tree there should only be one weight
        for node in span_data.nodes_to_date:
            assert len(span_data.get_weights(node)), 1
        for nd, expd_tips in [
            (6, 4),  # Node 6 (root) expected to have 4 descendant tips
            (5, 3),  # Node 5 (1st internal node) expected to have 3 descendant tips
            (4, 2),
        ]:  # Node 4 (2nd internal node) expected to have 3 descendant tips
            assert np.isin(span_data.get_weights(nd)[n]["descendant_tips"], expd_tips)

    def test_two_trees(self):
        ts = utility_functions.two_tree_ts()
        n = ts.num_samples
        span_data = self.verify_weights(ts)
        assert span_data.lookup_weight(5, n, 3) == 1.0  # Root on R tree
        assert span_data.lookup_weight(4, n, 3) == 0.2  # Root on L tree ...
        # ... but internal node on R tree
        assert span_data.lookup_weight(4, n, 2) == 0.8
        assert span_data.lookup_weight(3, n, 2) == 1.0  # Internal nd on L tree

    def test_missing_tree(self):
        ts = utility_functions.two_tree_ts().keep_intervals([(0, 0.2)], simplify=False)
        n = ts.num_samples
        # Here we have no reference in the trees to node 5
        with pytest.raises(ValueError, match="nodes not in any tree"):
            SpansBySamples(ts)
        ts = ts.simplify()
        span_data = self.verify_weights(ts)
        # Root on (deleted) R tree is missing
        assert 5 not in span_data.nodes_to_date
        assert span_data.lookup_weight(4, n, 3) == 1.0  # Root on L tree ...
        # ... but internal on (deleted) R tree
        assert not np.isin(span_data.get_weights(4)[n]["descendant_tips"], 2)
        assert span_data.lookup_weight(3, n, 2) == 1.0  # Internal nd on L tree

    def test_tree_with_unary_nodes(self):
        ts = utility_functions.single_tree_ts_with_unary()
        with pytest.raises(ValueError, match="unary"):
            self.verify_weights(ts)

    @pytest.mark.skip("Unary node is internal then the oldest node")
    def test_tree_with_unary_nodes_oldest(self):
        ts = utility_functions.two_tree_ts_with_unary_n3()
        n = ts.num_samples
        span_data = self.verify_weights(ts)
        assert span_data.lookup_weight(9, n, 4) == 0.5
        assert span_data.lookup_weight(8, n, 4) == 1.0
        assert span_data.lookup_weight(7, n, 1) == 0.5
        assert span_data.lookup_weight(7, n, 4) == 0.5
        assert span_data.lookup_weight(6, n, 2) == 0.5
        assert span_data.lookup_weight(6, n, 4) == 0.5
        assert span_data.lookup_weight(5, n, 2) == 0.5
        assert span_data.lookup_weight(4, n, 2) == 1.0

    def test_polytomy_tree(self):
        ts = utility_functions.polytomy_tree_ts()
        span_data = self.verify_weights(ts)
        assert span_data.lookup_weight(3, ts.num_samples, 3) == 1.0

    def test_larger_find_node_tip_weights(self):
        ts = msprime.simulate(
            10, recombination_rate=5, mutation_rate=5, random_seed=123
        )
        assert ts.num_trees > 1
        self.verify_weights(ts)

    def test_dangling_nodes_error(self):
        ts = utility_functions.single_tree_ts_n2_dangling()
        with pytest.raises(ValueError, match="dangling"):
            self.verify_weights(ts)

    def test_single_tree_n2_delete_intervals(self):
        ts = utility_functions.single_tree_ts_n2()
        deleted_interval_ts = ts.delete_intervals([[0.5, 0.6]])
        n = deleted_interval_ts.num_samples
        span_data = self.verify_weights(ts)
        span_data_deleted = self.verify_weights(deleted_interval_ts)
        assert span_data.lookup_weight(2, n, 2) == span_data_deleted.lookup_weight(
            2, n, 2
        )

    def test_single_tree_n4_delete_intervals(self):
        ts = utility_functions.single_tree_ts_n4()
        deleted_interval_ts = ts.delete_intervals([[0.5, 0.6]])
        n = deleted_interval_ts.num_samples
        span_data = self.verify_weights(ts)
        span_data_deleted = self.verify_weights(deleted_interval_ts)
        assert span_data.lookup_weight(4, n, 2) == span_data_deleted.lookup_weight(
            4, n, 2
        )
        assert span_data.lookup_weight(5, n, 3) == span_data_deleted.lookup_weight(
            5, n, 3
        )
        assert span_data.lookup_weight(6, n, 4) == span_data_deleted.lookup_weight(
            6, n, 4
        )

    def test_two_tree_ts_delete_intervals(self):
        ts = utility_functions.two_tree_ts()
        deleted_interval_ts = ts.delete_intervals([[0.5, 0.6]])
        n = deleted_interval_ts.num_samples
        span_data = self.verify_weights(ts)
        span_data_deleted = self.verify_weights(deleted_interval_ts)
        assert span_data.lookup_weight(3, n, 2) == span_data_deleted.lookup_weight(
            3, n, 2
        )
        assert pytest.approx(span_data_deleted.lookup_weight(4, n, 2)[0], 0.7 / 0.9)
        assert pytest.approx(span_data_deleted.lookup_weight(4, n, 3)[0], 0.2 / 0.9)
        assert span_data.lookup_weight(5, n, 3) == span_data_deleted.lookup_weight(
            3, n, 2
        )

    @pytest.mark.skip("YAN to fix")
    def test_truncated_nodes(self):
        Ne = 1e2
        ts = msprime.simulate(
            10, Ne=Ne, length=400, recombination_rate=1e-4, random_seed=12
        )
        truncated_ts = utility_functions.truncate_ts_samples(
            ts, average_span=200, random_seed=123
        )
        span_data = self.verify_weights(truncated_ts)
        raise NotImplementedError(str(span_data))


class TestMakePrior:
    # We only test make_prior() on single trees
    def verify_priors(self, ts, prior_distr):
        # Check prior contains all possible tips
        priors = ConditionalCoalescentTimes(None, Ne=0.5, prior_distr=prior_distr)
        priors.add(ts.num_samples)
        priors_df = priors[ts.num_samples]
        assert priors_df.shape[0] == ts.num_samples + 1
        return priors_df

    def test_one_tree_n2(self):
        ts = utility_functions.single_tree_ts_n2()
        priors = self.verify_priors(ts, "gamma")
        assert np.allclose(
            priors[2], PriorParams(alpha=1.0, beta=1.0, mean=1.0, var=1.0)
        )
        priors = self.verify_priors(ts, "lognorm")
        assert np.allclose(
            priors[2],
            PriorParams(alpha=-0.34657359, beta=0.69314718, mean=1.0, var=1.0),
        )

    def test_one_tree_n3(self):
        ts = utility_functions.single_tree_ts_n3()
        prior2mv = {"mean": 1 / 3, "var": 1 / 9}
        prior3mv = {"mean": 1 + 1 / 3, "var": 1 + 1 / 9}
        priors = self.verify_priors(ts, "lognorm")
        assert np.allclose(
            priors[2], PriorParams(alpha=-1.44518588, beta=0.69314718, **prior2mv)
        )
        assert np.allclose(
            priors[3], PriorParams(alpha=0.04492816, beta=0.48550782, **prior3mv)
        )
        priors = self.verify_priors(ts, "gamma")
        assert np.allclose(priors[2], PriorParams(alpha=1.0, beta=3.0, **prior2mv))
        assert np.allclose(priors[3], PriorParams(alpha=1.6, beta=1.2, **prior3mv))

    @pytest.mark.skip("Fill in values instead of np.nan")
    def test_one_tree_n4(self):
        ts = utility_functions.single_tree_ts_n4()
        prior2mv = {"mean": np.nan, "var": np.nan}
        prior3mv = {"mean": np.nan, "var": np.nan}
        prior4mv = {"mean": np.nan, "var": np.nan}

        priors = self.verify_priors(ts, "lognorm")
        assert np.allclose(
            priors[2], PriorParams(alpha=np.nan, beta=np.nan, **prior2mv)
        )
        assert np.allclose(
            priors[3], PriorParams(alpha=np.nan, beta=np.nan, **prior3mv)
        )
        assert np.allclose(
            priors[4], PriorParams(alpha=np.nan, beta=np.nan, **prior4mv)
        )

        priors = self.verify_priors(ts, "gamma")
        assert np.allclose(
            priors[2], PriorParams(alpha=np.nan, beta=np.nan, **prior2mv)
        )
        assert np.allclose(
            priors[3], PriorParams(alpha=np.nan, beta=np.nan, **prior3mv)
        )
        assert np.allclose(
            priors[4], PriorParams(alpha=np.nan, beta=np.nan, **prior4mv)
        )

    @pytest.mark.skip("Fill in values instead of np.nan")
    def test_polytomy_tree(self):
        ts = utility_functions.polytomy_tree_ts()
        prior3mv = {"mean": np.nan, "var": np.nan}

        priors = self.verify_priors(ts, "lognorm")
        assert np.allclose(
            priors[3], PriorParams(alpha=np.nan, beta=np.nan, **prior3mv)
        )

        priors = self.verify_prior(ts, "gamma")
        assert np.allclose(
            priors[3], PriorParams(alpha=np.nan, beta=np.nan, **prior3mv)
        )

    @pytest.mark.skip("Fill in values instead of np.nan")
    def test_two_tree_ts(self):
        ts = utility_functions.two_tree_ts()
        prior2mv = {"mean": np.nan, "var": np.nan}
        prior3mv = {"mean": np.nan, "var": np.nan}

        priors = self.verify_priors(ts, "lognorm")
        assert np.allclose(
            priors[2], PriorParams(alpha=np.nan, beta=np.nan, **prior2mv)
        )
        assert np.allclose(
            priors[3], PriorParams(alpha=np.nan, beta=np.nan, **prior3mv)
        )

        priors = self.verify_priors(ts, "gamma")
        assert np.allclose(
            priors[2], PriorParams(alpha=np.nan, beta=np.nan, **prior2mv)
        )
        assert np.allclose(
            priors[3], PriorParams(alpha=np.nan, beta=np.nan, **prior3mv)
        )

    @pytest.mark.skip("Fill in values instead of np.nan")
    def test_single_tree_ts_with_unary(self):
        ts = utility_functions.single_tree_ts_with_unary()
        prior2mv = {"mean": np.nan, "var": np.nan}
        prior3mv = {"mean": np.nan, "var": np.nan}

        priors = self.verify_priors(ts, "lognorm")
        assert np.allclose(
            priors[2], PriorParams(alpha=np.nan, beta=np.nan, **prior2mv)
        )
        assert np.allclose(
            priors[3], PriorParams(alpha=np.nan, beta=np.nan, **prior3mv)
        )

        priors = self.verify_priors(ts, "gamma")
        assert np.allclose(priors[2], PriorParams(alpha=1.0, beta=3.0, **prior2mv))
        assert np.allclose(priors[3], PriorParams(alpha=1.6, beta=1.2, **prior3mv))

    @pytest.mark.skip("Fill in values instead of np.nan")
    def test_two_tree_mutation_ts(self):
        ts = utility_functions.two_tree_mutation_ts()
        prior2mv = {"mean": np.nan, "var": np.nan}
        prior3mv = {"mean": np.nan, "var": np.nan}

        priors = self.verify_priors(ts, "lognorm")
        assert np.allclose(
            priors[2], PriorParams(alpha=np.nan, beta=np.nan, **prior2mv)
        )
        assert np.allclose(
            priors[3], PriorParams(alpha=np.nan, beta=np.nan, **prior3mv)
        )

        priors = self.verify_priors(ts, "gamma")
        assert np.allclose(priors[2], PriorParams(alpha=1.0, beta=3.0, **prior2mv))
        assert np.allclose(priors[3], PriorParams(alpha=1.6, beta=1.2, **prior3mv))


class TestMixturePrior:
    alpha_beta = [PriorParams.field_index("alpha"), PriorParams.field_index("beta")]

    def get_mixture_prior_params(self, ts, prior_distr):
        span_data = SpansBySamples(ts)
        priors = ConditionalCoalescentTimes(None, Ne=0.5, prior_distr=prior_distr)
        priors.add(ts.num_samples, approximate=False)
        mixture_priors = priors.get_mixture_prior_params(span_data)
        return mixture_priors

    def test_one_tree_n2(self):
        ts = utility_functions.single_tree_ts_n2()
        mixture_priors = self.get_mixture_prior_params(ts, "gamma")
        assert np.allclose(mixture_priors[2, self.alpha_beta], [1.0, 1.0])
        mixture_priors = self.get_mixture_prior_params(ts, "lognorm")
        assert np.allclose(
            mixture_priors[2, self.alpha_beta], [-0.34657359, 0.69314718]
        )

    def test_one_tree_n3(self):
        ts = utility_functions.single_tree_ts_n3()
        mixture_priors = self.get_mixture_prior_params(ts, "gamma")
        assert np.allclose(mixture_priors[3, self.alpha_beta], [1.0, 3.0])
        assert np.allclose(mixture_priors[4, self.alpha_beta], [1.6, 1.2])
        mixture_priors = self.get_mixture_prior_params(ts, "lognorm")
        assert np.allclose(
            mixture_priors[3, self.alpha_beta], [-1.44518588, 0.69314718]
        )
        assert np.allclose(mixture_priors[4, self.alpha_beta], [0.04492816, 0.48550782])

    def test_one_tree_n4(self):
        ts = utility_functions.single_tree_ts_n4()
        mixture_priors = self.get_mixture_prior_params(ts, "gamma")
        assert np.allclose(mixture_priors[4, self.alpha_beta], [0.81818182, 3.27272727])
        assert np.allclose(mixture_priors[5, self.alpha_beta], [1.8, 3.6])
        assert np.allclose(mixture_priors[6, self.alpha_beta], [1.97560976, 1.31707317])

    def test_polytomy_tree(self):
        ts = utility_functions.polytomy_tree_ts()
        mixture_priors = self.get_mixture_prior_params(ts, "gamma")
        assert np.allclose(mixture_priors[3, self.alpha_beta], [1.6, 1.2])

    def test_two_trees(self):
        ts = utility_functions.two_tree_ts()
        mixture_priors = self.get_mixture_prior_params(ts, "gamma")
        assert np.allclose(mixture_priors[3, self.alpha_beta], [1.0, 3.0])
        # Node 4 should be a mixture between 2 and 3 tips
        assert np.allclose(mixture_priors[4, self.alpha_beta], [0.60377, 1.13207])
        assert np.allclose(mixture_priors[5, self.alpha_beta], [1.6, 1.2])

    def test_single_tree_ts_with_unary(self):
        ts = utility_functions.single_tree_ts_with_unary()
        with pytest.raises(ValueError, match="unary"):
            self.get_mixture_prior_params(ts, "gamma")
        # # Root is a 3 tip prior
        # assert np.allclose(mixture_priors[7, self.alpha_beta], [1.6, 1.2])
        # # Node 6 should be a 50:50 mixture between 1 and 3 tips
        # assert np.allclose(mixture_priors[6, self.alpha_beta], [0.44444, 0.66666])
        # # Node 5 should be a 50:50 mixture of 2 and 3 tips
        # assert np.allclose(mixture_priors[5, self.alpha_beta], [0.80645, 0.96774])
        # # Node 4 should be a 75:25 mixture of 2 and 3 tips
        # assert np.allclose(mixture_priors[4, self.alpha_beta], [0.62025, 1.06329])
        # # Node 3 is a 2 tip prior
        # assert np.allclose(mixture_priors[3, self.alpha_beta], [1.0, 3.0])

    def test_two_tree_mutation_ts(self):
        ts = utility_functions.two_tree_mutation_ts()
        mixture_priors = self.get_mixture_prior_params(ts, "gamma")
        assert np.allclose(mixture_priors[3, self.alpha_beta], [1.0, 3.0])
        # Node 4 should be a mixture between 2 and 3 tips
        assert np.allclose(mixture_priors[4, self.alpha_beta], [0.60377, 1.13207])
        assert np.allclose(mixture_priors[5, self.alpha_beta], [1.6, 1.2])

    def check_intervals(self, ts, delete_interval_ts, keep_interval_ts):
        tests = list()
        for distr in ["gamma", "lognorm"]:
            mix_priors = self.get_mixture_prior_params(ts, distr)
            for interval_ts in [delete_interval_ts, keep_interval_ts]:
                mix_priors_ints = self.get_mixture_prior_params(interval_ts, distr)
                for internal_node in range(ts.num_samples, ts.num_nodes):
                    tests.append(
                        np.allclose(
                            mix_priors[internal_node, self.alpha_beta],
                            mix_priors_ints[internal_node, self.alpha_beta],
                        )
                    )
        return tests

    def test_one_tree_n2_intervals(self):
        ts = utility_functions.single_tree_ts_n2()
        delete_interval_ts = ts.delete_intervals([[0.5, 0.6]])
        keep_interval_ts = ts.keep_intervals([[0, 0.1]])
        tests = self.check_intervals(ts, delete_interval_ts, keep_interval_ts)
        assert np.all(tests)

    def test_two_tree_mutation_ts_intervals(self):
        ts = utility_functions.two_tree_mutation_ts()
        ts_extra_length = utility_functions.two_tree_ts_extra_length()
        delete_interval_ts = ts_extra_length.delete_intervals([[0.75, 1.25]])
        keep_interval_ts = ts_extra_length.keep_intervals([[0, 1.0]])
        tests = self.check_intervals(ts, delete_interval_ts, keep_interval_ts)
        assert np.all(tests)


class TestPriorVals:
    def verify_prior_vals(self, ts, prior_distr):
        span_data = SpansBySamples(ts)
        Ne = 0.5
        priors = ConditionalCoalescentTimes(None, Ne=Ne, prior_distr=prior_distr)
        priors.add(ts.num_samples, approximate=False)
        grid = np.linspace(0, 3, 3)
        mixture_priors = priors.get_mixture_prior_params(span_data)
        prior_vals = fill_priors(mixture_priors, grid, ts, Ne, prior_distr=prior_distr)
        return prior_vals

    def test_one_tree_n2(self):
        ts = utility_functions.single_tree_ts_n2()
        prior_vals = self.verify_prior_vals(ts, "gamma")
        assert np.allclose(prior_vals[2], [0, 1, 0.22313016])

    def test_one_tree_n3(self):
        ts = utility_functions.single_tree_ts_n3()
        prior_vals = self.verify_prior_vals(ts, "gamma")
        assert np.allclose(prior_vals[3], [0, 1, 0.011109])
        assert np.allclose(prior_vals[4], [0, 1, 0.3973851])

    def test_one_tree_n4(self):
        ts = utility_functions.single_tree_ts_n4()
        prior_vals = self.verify_prior_vals(ts, "gamma")
        assert np.allclose(prior_vals[4], [0, 1, 0.00467134])
        assert np.allclose(prior_vals[5], [0, 1, 0.02167806])
        assert np.allclose(prior_vals[6], [0, 1, 0.52637529])

    def test_polytomy_tree(self):
        ts = utility_functions.polytomy_tree_ts()
        prior_vals = self.verify_prior_vals(ts, "gamma")
        assert np.allclose(prior_vals[3], [0, 1, 0.3973851])

    def test_two_tree_ts(self):
        ts = utility_functions.two_tree_ts()
        prior_vals = self.verify_prior_vals(ts, "gamma")
        assert np.allclose(prior_vals[3], [0, 1, 0.011109])
        assert np.allclose(prior_vals[4], [0, 1, 0.080002])
        assert np.allclose(prior_vals[5], [0, 1, 0.3973851])

    def test_tree_with_unary_nodes(self):
        ts = utility_functions.single_tree_ts_with_unary()
        with pytest.raises(ValueError, match="unary"):
            self.verify_prior_vals(ts, "gamma")
        # assert np.allclose(prior_vals[7], [0, 1, 0.397385])
        # assert np.allclose(prior_vals[6], [0, 1, 0.113122])
        # assert np.allclose(prior_vals[5], [0, 1, 0.164433])
        # assert np.allclose(prior_vals[4], [0, 1, 0.093389])
        # assert np.allclose(prior_vals[3], [0, 1, 0.011109])

    def test_one_tree_n2_intervals(self):
        ts = utility_functions.single_tree_ts_n2()
        delete_interval_ts = ts.delete_intervals([[0.1, 0.3]])
        keep_interval_ts = ts.keep_intervals([[0.4, 0.6]])
        prior_vals = self.verify_prior_vals(ts, "gamma")
        prior_vals_keep = self.verify_prior_vals(keep_interval_ts, "gamma")
        prior_vals_delete = self.verify_prior_vals(delete_interval_ts, "gamma")
        assert np.allclose(prior_vals[2], prior_vals_keep[2])
        assert np.allclose(prior_vals[2], prior_vals_delete[2])


class TestLikelihoodClass:
    def poisson(self, param, x, normalize=True):
        ll = np.exp(-param) * param ** x / scipy.special.factorial(x)
        if normalize:
            return ll / np.max(ll)
        else:
            return ll

    def log_poisson(self, param, x, normalize=True):
        with np.errstate(divide="ignore"):
            ll = np.log(np.exp(-param) * param ** x / scipy.special.factorial(x))
        if normalize:
            return ll - np.max(ll)
        else:
            return ll

    def test_get_mut_edges(self):
        ts = utility_functions.two_tree_mutation_ts()
        mutations_per_edge = Likelihoods.get_mut_edges(ts)
        for e in ts.edges():
            if e.child == 3 and e.parent == 4:
                assert mutations_per_edge[e.id] == 2
            elif e.child == 0 and e.parent == 5:
                assert mutations_per_edge[e.id] == 1
            else:
                assert mutations_per_edge[e.id] == 0

    def test_create_class(self):
        ts = utility_functions.two_tree_mutation_ts()
        grid = np.array([0, 1, 2])
        lik = Likelihoods(ts, grid)
        loglik = LogLikelihoods(ts, grid)
        with pytest.raises(AssertionError):
            lik.get_mut_lik_fixed_node(ts.edge(0))
        with pytest.raises(AssertionError):
            lik.get_mut_lik_lower_tri(ts.edge(0))
        with pytest.raises(AssertionError):
            lik.get_mut_lik_upper_tri(ts.edge(0))
        with pytest.raises(AssertionError):
            loglik.get_mut_lik_fixed_node(ts.edge(0))
        with pytest.raises(AssertionError):
            loglik.get_mut_lik_lower_tri(ts.edge(0))
        with pytest.raises(AssertionError):
            loglik.get_mut_lik_upper_tri(ts.edge(0))

    def test_no_theta_class(self):
        ts = utility_functions.two_tree_mutation_ts()
        grid = np.array([0, 1, 2])
        lik = Likelihoods(ts, grid, mutation_rate=None)
        with pytest.raises(RuntimeError):
            lik.precalculate_mutation_likelihoods()

    def test_precalc_lik_lower(self):
        ts = utility_functions.single_tree_ts_n3()
        grid = np.array([0, 1, 2])
        eps = 0
        mut_rate = 1
        lik = Likelihoods(ts, grid, mut_rate, eps)
        for method in (0, 1, 2):
            # TODO: Remove this loop and hard-code one of the methods after perf testing
            lik.precalculate_mutation_likelihoods(unique_method=method)
            assert ts.num_trees == 1
            span = ts.first().span
            dt = grid
            num_muts = 0
            n_internal_edges = 0
            expected_lik_dt = self.poisson(dt * (mut_rate * span), num_muts)
            for edge in ts.edges():
                if ts.node(edge.child).is_sample():
                    with pytest.raises(AssertionError):
                        lik.get_mut_lik_lower_tri(edge)
                    with pytest.raises(AssertionError):
                        lik.get_mut_lik_upper_tri(edge)
                    fixed_edge_lik = lik.get_mut_lik_fixed_node(edge)
                    assert np.allclose(fixed_edge_lik, expected_lik_dt)
                else:
                    n_internal_edges += 1  # only one internal edge in this tree
                    assert n_internal_edges <= 1
                    with pytest.raises(AssertionError):
                        lik.get_mut_lik_fixed_node(edge)
                    lower_tri = lik.get_mut_lik_lower_tri(edge)

                    assert pytest.approx(lower_tri[0], expected_lik_dt[0])

                    assert pytest.approx(lower_tri[1], expected_lik_dt[1])
                    assert pytest.approx(lower_tri[2], expected_lik_dt[0])

                    assert pytest.approx(lower_tri[3], expected_lik_dt[2])
                    assert pytest.approx(lower_tri[4], expected_lik_dt[1])
                    assert pytest.approx(lower_tri[5], expected_lik_dt[0])

    def test_precalc_lik_upper_multithread(self):
        ts = utility_functions.two_tree_mutation_ts()
        grid = np.array([0, 1, 2])
        eps = 0
        mut_rate = 1
        for L, pois in [
            (Likelihoods, self.poisson),
            (LogLikelihoods, self.log_poisson),
        ]:
            for normalize in (True, False):
                lik = L(ts, grid, mut_rate, eps, normalize=normalize)
                dt = grid
                for num_threads in (None, 1, 2):
                    n_internal_edges = 0
                    lik.precalculate_mutation_likelihoods(num_threads=num_threads)
                    for edge in ts.edges():
                        if not ts.node(edge.child).is_sample():
                            n_internal_edges += (
                                1  # only two internal edges in this tree
                            )
                            assert n_internal_edges <= 2
                            if edge.parent == 4 and edge.child == 3:
                                num_muts = 2
                            elif edge.parent == 5 and edge.child == 4:
                                num_muts = 0
                            else:
                                self.fail("Unexpected edge")
                            span = edge.right - edge.left
                            expected_lik_dt = pois(
                                dt * (mut_rate * span),
                                num_muts,
                                normalize=normalize,
                            )
                            upper_tri = lik.get_mut_lik_upper_tri(edge)

                            assert pytest.approx(upper_tri[0], expected_lik_dt[0])
                            assert pytest.approx(upper_tri[1], expected_lik_dt[1])
                            assert pytest.approx(upper_tri[2], expected_lik_dt[2])

                            assert pytest.approx(upper_tri[3], expected_lik_dt[0])
                            assert pytest.approx(upper_tri[4], expected_lik_dt[1])

                            assert pytest.approx(upper_tri[5], expected_lik_dt[0])

    def test_tri_functions(self):
        ts = utility_functions.two_tree_mutation_ts()
        grid = np.array([0, 1, 2])
        eps = 0
        mut_rate = 1
        lik = Likelihoods(ts, grid, mut_rate, eps)
        lik.precalculate_mutation_likelihoods()
        for e in ts.edges():
            if e.child == 3 and e.parent == 4:
                exp_branch_muts = 2
                exp_span = 0.2
                assert e.right - e.left == exp_span
                assert lik.mut_edges[e.id] == exp_branch_muts
                pois_lambda = grid * mut_rate * exp_span
                cumul_pois = np.cumsum(self.poisson(pois_lambda, exp_branch_muts))
                lower_tri = lik.get_mut_lik_lower_tri(e)
                assert np.allclose(lik.rowsum_lower_tri(lower_tri), cumul_pois)
                upper_tri = lik.get_mut_lik_upper_tri(e)
                assert np.allclose(lik.rowsum_upper_tri(upper_tri)[::-1], cumul_pois)

    def test_no_theta_class_loglikelihood(self):
        ts = utility_functions.two_tree_mutation_ts()
        grid = np.array([0, 1, 2])
        lik = LogLikelihoods(ts, grid, mutation_rate=None)
        with pytest.raises(RuntimeError):
            lik.precalculate_mutation_likelihoods()

    @staticmethod
    def naive_logsumexp(X):
        r = 0
        for x in X:
            r += np.exp(x)
        return np.log(r)

    def test_logsumexp(self):
        lls = np.array([0.1, 0.2, 0.5])
        ll_sum = np.sum(lls)
        log_lls = np.log(lls)
        assert np.allclose(LogLikelihoods.logsumexp(log_lls), np.log(ll_sum))

    def test_zeros_logsumexp(self):
        lls = np.log(np.concatenate([np.zeros(100), np.random.rand(1000)]))
        assert np.allclose(LogLikelihoods.logsumexp(lls), self.naive_logsumexp(lls))

    def test_logsumexp_underflow(self):
        # underflow in the naive case, but not in the LogLikelihoods implementation
        lls = np.array([-1000, -1001])
        assert self.naive_logsumexp(lls) == -np.inf
        assert LogLikelihoods.logsumexp(lls) != -np.inf

    def test_log_tri_functions(self):
        ts = utility_functions.two_tree_mutation_ts()
        grid = np.array([0, 1, 2])
        eps = 0
        mut_rate = 1
        lik = Likelihoods(ts, grid, mut_rate, eps)
        loglik = LogLikelihoods(ts, grid, mutation_rate=mut_rate, eps=eps)
        lik.precalculate_mutation_likelihoods()
        loglik.precalculate_mutation_likelihoods()
        for e in ts.edges():
            if e.child == 3 and e.parent == 4:
                exp_branch_muts = 2
                exp_span = 0.2
                assert e.right - e.left == exp_span
                assert lik.mut_edges[e.id] == exp_branch_muts
                assert loglik.mut_edges[e.id] == exp_branch_muts
                pois_lambda = grid * mut_rate * exp_span
                cumul_pois = np.cumsum(self.poisson(pois_lambda, exp_branch_muts))
                lower_tri = lik.get_mut_lik_lower_tri(e)
                lower_tri_log = loglik.get_mut_lik_lower_tri(e)
                assert np.allclose(lik.rowsum_lower_tri(lower_tri), cumul_pois)
                with np.errstate(divide="ignore"):
                    assert np.allclose(
                        loglik.rowsum_lower_tri(lower_tri_log), np.log(cumul_pois)
                    )
                upper_tri = lik.get_mut_lik_upper_tri(e)
                upper_tri_log = loglik.get_mut_lik_upper_tri(e)
                assert np.allclose(lik.rowsum_upper_tri(upper_tri)[::-1], cumul_pois)
                with np.errstate(divide="ignore"):
                    assert np.allclose(
                        loglik.rowsum_upper_tri(upper_tri_log)[::-1],
                        np.log(cumul_pois),
                    )


class TestNodeGridValuesClass:
    # TODO - needs a few more tests in here
    def test_init(self):
        num_nodes = 5
        ids = np.array([3, 4])
        timepoints = np.array(range(10))
        store = NodeGridValues(num_nodes, ids, timepoints, fill_value=6)
        assert store.grid_data.shape == (len(ids), len(timepoints))
        assert len(store.fixed_data) == (num_nodes - len(ids))
        assert np.all(store.grid_data == 6)
        assert np.all(store.fixed_data == 6)

        ids = np.array([3, 4], dtype=np.int32)
        store = NodeGridValues(num_nodes, ids, timepoints, fill_value=5)
        assert store.grid_data.shape == (len(ids), len(timepoints))
        assert len(store.fixed_data) == num_nodes - len(ids)
        assert np.all(store.fixed_data == 5)

    def test_set_and_get(self):
        num_nodes = 5
        grid_size = 2
        fill = {}
        for ids in ([3, 4], []):
            np.random.seed(1)
            store = NodeGridValues(
                num_nodes, np.array(ids, dtype=np.int32), np.array(range(grid_size))
            )
            for i in range(num_nodes):
                fill[i] = np.random.random(grid_size if i in ids else None)
                store[i] = fill[i]
            for i in range(num_nodes):
                assert np.all(fill[i] == store[i])
        with pytest.raises(IndexError):
            store.__getitem__(num_nodes)

    def test_bad_init(self):
        ids = [3, 4]
        with pytest.raises(ValueError):
            NodeGridValues(3, np.array(ids), np.array([0, 1.2, 2]))
        with pytest.raises(AttributeError):
            NodeGridValues(5, np.array(ids), -1)
        with pytest.raises(ValueError):
            NodeGridValues(5, np.array([-1]), np.array([0, 1.2, 2]))

    def test_clone(self):
        num_nodes = 10
        grid_size = 2
        ids = [3, 4]
        orig = NodeGridValues(num_nodes, np.array(ids), np.array(range(grid_size)))
        orig[3] = np.array([1, 2])
        orig[4] = np.array([4, 3])
        orig[0] = 1.5
        orig[9] = 2.5
        # test with np.zeros
        clone = NodeGridValues.clone_with_new_data(orig, 0)
        assert clone.grid_data.shape == orig.grid_data.shape
        assert clone.fixed_data.shape == orig.fixed_data.shape
        assert np.all(clone.grid_data == 0)
        assert np.all(clone.fixed_data == 0)
        # test with something else
        clone = NodeGridValues.clone_with_new_data(orig, 5)
        assert clone.grid_data.shape == orig.grid_data.shape
        assert clone.fixed_data.shape == orig.fixed_data.shape
        assert np.all(clone.grid_data == 5)
        assert np.all(clone.fixed_data == 5)
        # test with different
        scalars = np.arange(num_nodes - len(ids))
        clone = NodeGridValues.clone_with_new_data(orig, 0, scalars)
        assert clone.grid_data.shape == orig.grid_data.shape
        assert clone.fixed_data.shape == orig.fixed_data.shape
        assert np.all(clone.grid_data == 0)
        assert np.all(clone.fixed_data == scalars)

        clone = NodeGridValues.clone_with_new_data(orig, np.array([[1, 2], [4, 3]]))
        for i in range(num_nodes):
            if i in ids:
                assert np.all(clone[i] == orig[i])
            else:
                assert np.isnan(clone[i])
        clone = NodeGridValues.clone_with_new_data(orig, np.array([[1, 2], [4, 3]]), 0)
        for i in range(num_nodes):
            if i in ids:
                assert np.all(clone[i] == orig[i])
            else:
                assert clone[i] == 0

    def test_bad_clone(self):
        num_nodes = 10
        ids = [3, 4]
        orig = NodeGridValues(num_nodes, np.array(ids), np.array([0, 1.2]))
        with pytest.raises(ValueError):
            NodeGridValues.clone_with_new_data(
                orig,
                np.array([[1, 2, 3], [4, 5, 6]]),
            )
        with pytest.raises(ValueError):
            NodeGridValues.clone_with_new_data(
                orig,
                0,
                np.array([[1, 2], [4, 5]]),
            )


class TestAlgorithmClass:
    def test_nonmatching_prior_vs_lik_timepoints(self):
        ts = utility_functions.single_tree_ts_n3()
        timepoints1 = np.array([0, 1.2, 2])
        timepoints2 = np.array([0, 1.1, 2])
        Ne = 0.5
        priors = tsdate.build_prior_grid(ts, Ne, timepoints1)
        lls = Likelihoods(ts, timepoints2)
        with pytest.raises(ValueError, match="timepoints"):
            InOutAlgorithms(priors, lls)

    def test_nonmatching_prior_vs_lik_fixednodes(self):
        ts1 = utility_functions.single_tree_ts_n3()
        ts2 = utility_functions.single_tree_ts_n2_dangling()
        timepoints = np.array([0, 1.2, 2])
        Ne = 0.5
        priors = tsdate.build_prior_grid(ts1, Ne, timepoints)
        lls = Likelihoods(ts2, priors.timepoints)
        with pytest.raises(ValueError, match="fixed"):
            InOutAlgorithms(priors, lls)


class TestInsideAlgorithm:
    def run_inside_algorithm(self, ts, prior_distr, normalize=True):
        Ne = 0.5
        priors = tsdate.build_prior_grid(
            ts,
            Ne,
            timepoints=np.array([0, 1.2, 2]),
            approximate_priors=False,
            prior_distribution=prior_distr,
        )
        eps = 1e-6
        mut_rate = 0.5
        lls = Likelihoods(ts, priors.timepoints, mut_rate, eps=eps)
        lls.precalculate_mutation_likelihoods()
        algo = InOutAlgorithms(priors, lls)
        algo.inside_pass(normalize=normalize)
        return algo, priors

    def test_one_tree_n2(self):
        ts = utility_functions.single_tree_ts_n2()
        algo = self.run_inside_algorithm(ts, "gamma")[0]
        assert np.allclose(algo.inside[2], np.array([0, 1, 0.10664654]))

    def test_one_tree_n3(self):
        ts = utility_functions.single_tree_ts_n3()
        algo = self.run_inside_algorithm(ts, "gamma")[0]
        assert np.allclose(algo.inside[3], np.array([0, 1, 0.0114771635]))
        assert np.allclose(algo.inside[4], np.array([0, 1, 0.1941815518]))

    def test_one_tree_n4(self):
        ts = utility_functions.single_tree_ts_n4()
        algo = self.run_inside_algorithm(ts, "gamma")[0]
        assert np.allclose(algo.inside[4], np.array([0, 1, 0.00548801]))
        assert np.allclose(algo.inside[5], np.array([0, 1, 0.0239174]))
        assert np.allclose(algo.inside[6], np.array([0, 1, 0.26222197]))

    def test_polytomy_tree(self):
        ts = utility_functions.polytomy_tree_ts()
        algo = self.run_inside_algorithm(ts, "gamma")[0]
        assert np.allclose(algo.inside[3], np.array([0, 1, 0.12797265]))

    def test_two_tree_ts(self):
        ts = utility_functions.two_tree_ts()
        algo, priors = self.run_inside_algorithm(ts, "gamma", normalize=False)
        mut_rate = 0.5
        # priors[3][1] * Ll_(0->3)(1.2 - 0 + eps) ** 2
        node3_t1 = (
            priors[3][1]
            * scipy.stats.poisson.pmf(0, (1.2 + 1e-6) * mut_rate * 0.2) ** 2
        )
        # priors[3][2] * sum(Ll_(0->3)(2 - t + eps))
        node3_t2 = (
            priors[3][2] * scipy.stats.poisson.pmf(0, (2 + 1e-6) * mut_rate * 0.2) ** 2
        )
        assert np.allclose(algo.inside[3], np.array([0, node3_t1, node3_t2]))
        """
        priors[4][1] * (Ll_(2->4)(1.2 - 0 + eps) * (Ll_(1->4)(1.2 - 0 + eps)) *
        (Ll_(3->4)(1.2-1.2+eps) * node3_t1)
        """
        node4_t1 = priors[4][1] * (
            scipy.stats.poisson.pmf(0, (1.2 + 1e-6) * mut_rate * 1)
            * scipy.stats.poisson.pmf(0, (1.2 + 1e-6) * mut_rate * 0.8)
            * (scipy.stats.poisson.pmf(0, (1e-6) * mut_rate * 0.2) * node3_t1)
        )
        """
        priors[4][2] * (Ll_(2->4)(2 - 0 + eps) * Ll_(1->4)(2 - 0 + eps) *
        (sum_(t'<2)(Ll_(3->4)(2-t'+eps) * node3_t))
        """
        node4_t2 = priors[4][2] * (
            scipy.stats.poisson.pmf(0, (2 + 1e-6) * mut_rate * 1)
            * scipy.stats.poisson.pmf(0, (2 + 1e-6) * mut_rate * 0.8)
            * (
                (scipy.stats.poisson.pmf(0, (0.8 + 1e-6) * mut_rate * 0.2) * node3_t1)
                + (
                    scipy.stats.poisson.pmf(0, (1e-6 + 1e-6) * mut_rate * 0.2)
                    * node3_t2
                )
            )
        )
        assert np.allclose(algo.inside[4], np.array([0, node4_t1, node4_t2]))
        """
        priors[5][1] * (Ll_(4->5)(1.2 - 1.2 + eps) * (node3_t ** 0.8)) *
        (Ll_(0->5)(1.2 - 0 + eps) * 1)
        raising node4_t to 0.8 is geometric scaling
        """
        node5_t1 = (
            priors[5][1]
            * (scipy.stats.poisson.pmf(0, (1e-6) * mut_rate * 0.8) * (node4_t1 ** 0.8))
            * (scipy.stats.poisson.pmf(0, (1.2 + 1e-6) * mut_rate * 0.8))
        )
        """
        prior[5][2] * (sum_(t'<1.2)(Ll_(4->5)(1.2 - 0 + eps) * (node3_t ** 0.8)) *
        (Ll_(0->5)(1.2 - 0 + eps) * 1)
        """
        node5_t2 = (
            priors[5][2]
            * (
                (
                    scipy.stats.poisson.pmf(0, (0.8 + 1e-6) * mut_rate * 0.8)
                    * (node4_t1 ** 0.8)
                )
                + (
                    scipy.stats.poisson.pmf(0, (1e-6 + 1e-6) * mut_rate * 0.8)
                    * (node4_t2 ** 0.8)
                )
            )
            * (scipy.stats.poisson.pmf(0, (2 + 1e-6) * mut_rate * 0.8))
        )
        assert np.allclose(algo.inside[5], np.array([0, node5_t1, node5_t2]))

    def test_tree_with_unary_nodes(self):
        ts = utility_functions.single_tree_ts_with_unary()
        with pytest.raises(ValueError, match="unary"):
            self.run_inside_algorithm(ts, "gamma")[0]
        # assert np.allclose(algo.inside[7], np.array([0, 1, 0.25406637]))
        # assert np.allclose(algo.inside[6], np.array([0, 1, 0.07506923]))
        # assert np.allclose(algo.inside[5], np.array([0, 1, 0.13189998]))
        # assert np.allclose(algo.inside[4], np.array([0, 1, 0.07370801]))
        # assert np.allclose(algo.inside[3], np.array([0, 1, 0.01147716]))

    def test_two_tree_mutation_ts(self):
        ts = utility_functions.two_tree_mutation_ts()
        algo = self.run_inside_algorithm(ts, "gamma")[0]
        assert np.allclose(algo.inside[3], np.array([0, 1, 0.02176622]))
        # self.assertTrue(np.allclose(upward[4], np.array([0, 2.90560754e-05, 1])))
        # NB the replacement below has not been hand-calculated
        assert np.allclose(algo.inside[4], np.array([0, 3.63200499e-11, 1]))
        # self.assertTrue(np.allclose(upward[5], np.array([0, 5.65044738e-05, 1])))
        # NB the replacement below has not been hand-calculated
        assert np.allclose(algo.inside[5], np.array([0, 7.06320034e-11, 1]))

    def test_dangling_fails(self):
        ts = utility_functions.single_tree_ts_n2_dangling()
        print(ts.draw_text())
        print("Samples:", ts.samples())
        Ne = 0.5
        with pytest.raises(ValueError, match="simplified"):
            tsdate.build_prior_grid(ts, Ne, timepoints=np.array([0, 1.2, 2]))
        # mut_rate = 1
        # eps = 1e-6
        # lls = Likelihoods(ts, priors.timepoints, mut_rate, eps)
        # algo = InOutAlgorithms(priors, lls)
        # with pytest.raises(ValueError, match="dangling"):
        #     algo.inside_pass()


class TestOutsideAlgorithm:
    def run_outside_algorithm(
        self, ts, prior_distr="lognorm", normalize=False, ignore_oldest_root=False
    ):
        span_data = SpansBySamples(ts)
        Ne = 0.5
        priors = ConditionalCoalescentTimes(None, Ne, prior_distr)
        priors.add(ts.num_samples, approximate=False)
        grid = np.array([0, 1.2, 2])
        mixture_priors = priors.get_mixture_prior_params(span_data)
        prior_vals = fill_priors(mixture_priors, grid, ts, Ne, prior_distr=prior_distr)
        mut_rate = 1
        eps = 1e-6
        lls = Likelihoods(ts, grid, mut_rate, eps=eps)
        lls.precalculate_mutation_likelihoods()
        algo = InOutAlgorithms(prior_vals, lls)
        algo.inside_pass()
        algo.outside_pass(normalize=normalize, ignore_oldest_root=ignore_oldest_root)
        return algo

    def test_one_tree_n2(self):
        ts = utility_functions.single_tree_ts_n2()
        for prior_distr in ("lognorm", "gamma"):
            algo = self.run_outside_algorithm(ts, prior_distr)
            # Root, should this be 0,1,1 or 1,1,1
            assert np.array_equal(algo.outside[2], np.array([1, 1, 1]))

    def test_one_tree_n3(self):
        ts = utility_functions.single_tree_ts_n3()
        for prior_distr in ("lognorm", "gamma"):
            algo = self.run_outside_algorithm(ts, prior_distr)
            # self.assertTrue(np.allclose(
            #                  downward[3], np.array([0, 1, 0.33508884])))
            assert np.allclose(algo.outside[4], np.array([1, 1, 1]))
            # self.assertTrue(np.allclose(
            #      posterior[3], np.array([0, 0.99616886, 0.00383114])))
            # self.assertTrue(np.allclose(
            #                 posterior[4], np.array([0, 0.83739361, 0.16260639])))

    def test_one_tree_n4(self):
        ts = utility_functions.single_tree_ts_n4()
        for prior_distr in ("lognorm", "gamma"):
            algo = self.run_outside_algorithm(ts, prior_distr)
            # self.assertTrue(np.allclose(
            #                 downward[4], np.array([0, 1, 0.02187283])))
            # self.assertTrue(np.allclose(
            #                 downward[5], np.array([0, 1, 0.41703272])))
            # Root, should this be 0,1,1 or 1,1,1
            assert np.allclose(algo.outside[6], np.array([1, 1, 1]))

    def test_outside_before_inside_fails(self):
        ts = utility_functions.single_tree_ts_n2()
        Ne = 0.5
        priors = tsdate.build_prior_grid(ts, Ne)
        mut_rate = 1
        lls = Likelihoods(ts, priors.timepoints, mut_rate)
        lls.precalculate_mutation_likelihoods()
        algo = InOutAlgorithms(priors, lls)
        with pytest.raises(RuntimeError):
            algo.outside_pass()

    def test_normalize_outside(self):
        ts = msprime.simulate(
            50, Ne=10000, mutation_rate=1e-8, recombination_rate=1e-8, random_seed=12
        )
        normalize = self.run_outside_algorithm(ts, normalize=True)
        no_normalize = self.run_outside_algorithm(ts, normalize=False)
        assert np.allclose(
            normalize.outside.grid_data[:],
            (
                no_normalize.outside.grid_data[:]
                / np.max(no_normalize.outside.grid_data[:], axis=1)[:, np.newaxis]
            ),
        )

    def test_ignore_oldest_root(self):
        ts = utility_functions.single_tree_ts_mutation_n3()
        ignore_oldest = self.run_outside_algorithm(ts, ignore_oldest_root=True)
        use_oldest = self.run_outside_algorithm(ts, ignore_oldest_root=False)
        assert ~np.array_equal(ignore_oldest.outside[3], use_oldest.outside[3])
        # When node is not used in outside algorithm, all values should be equal
        assert np.all(ignore_oldest.outside[3] == ignore_oldest.outside[3][0])
        assert np.all(use_oldest.outside[4] == use_oldest.outside[4][0])

    def test_ignore_oldest_root_two_mrcas(self):
        ts = utility_functions.two_tree_two_mrcas()
        ignore_oldest = self.run_outside_algorithm(ts, ignore_oldest_root=True)
        use_oldest = self.run_outside_algorithm(ts, ignore_oldest_root=False)
        assert ~np.array_equal(ignore_oldest.outside[7], use_oldest.outside[7])
        assert ~np.array_equal(ignore_oldest.outside[6], use_oldest.outside[6])
        # In this example, if the outside algorithm was *not* used, nodes 4 and 5 should
        # have same outside values. If it is used, node 5 should seem younger than 4
        assert np.array_equal(ignore_oldest.outside[4], ignore_oldest.outside[5])
        assert ~np.array_equal(use_oldest.outside[4], use_oldest.outside[5])


class TestTotalFunctionalValueTree:
    """
    Tests to ensure that we recover the total functional value of the tree.
    We can also recover this property in the tree sequence in the special case where
    all node times are known (or all bar one).
    """

    def find_posterior(self, ts, prior_distr):
        grid = np.array([0, 1.2, 2])
        span_data = SpansBySamples(ts)
        Ne = 0.5
        priors = ConditionalCoalescentTimes(None, Ne=Ne, prior_distr=prior_distr)
        priors.add(ts.num_samples, approximate=False)
        mixture_priors = priors.get_mixture_prior_params(span_data)
        prior_vals = fill_priors(mixture_priors, grid, ts, Ne, prior_distr=prior_distr)
        mut_rate = 1
        eps = 1e-6
        lls = Likelihoods(ts, grid, mut_rate, eps=eps)
        lls.precalculate_mutation_likelihoods()
        algo = InOutAlgorithms(prior_vals, lls)
        algo.inside_pass()
        posterior = algo.outside_pass(normalize=False)
        assert np.array_equal(
            np.sum(algo.inside.grid_data * algo.outside.grid_data, axis=1),
            np.sum(algo.inside.grid_data * algo.outside.grid_data, axis=1),
        )
        assert np.allclose(
            np.sum(algo.inside.grid_data * algo.outside.grid_data, axis=1),
            np.sum(algo.inside.grid_data[-1]),
        )
        return posterior, algo

    def test_one_tree_n2(self):
        ts = utility_functions.single_tree_ts_n2()
        for distr in ("gamma", "lognorm"):
            posterior, algo = self.find_posterior(ts, distr)

    def test_one_tree_n3(self):
        ts = utility_functions.single_tree_ts_n3()
        for distr in ("gamma", "lognorm"):
            posterior, algo = self.find_posterior(ts, distr)

    def test_one_tree_n4(self):
        ts = utility_functions.single_tree_ts_n4()
        for distr in ("gamma", "lognorm"):
            posterior, algo = self.find_posterior(ts, distr)

    def test_one_tree_n3_mutation(self):
        ts = utility_functions.single_tree_ts_mutation_n3()
        for distr in ("gamma", "lognorm"):
            posterior, algo = self.find_posterior(ts, distr)

    def test_polytomy_tree(self):
        ts = utility_functions.polytomy_tree_ts()
        for distr in ("gamma", "lognorm"):
            posterior, algo = self.find_posterior(ts, distr)

    def test_tree_with_unary_nodes(self):
        ts = utility_functions.single_tree_ts_with_unary()
        for distr in ("gamma", "lognorm"):
            with pytest.raises(ValueError, match="unary"):
                posterior, algo = self.find_posterior(ts, distr)


class TestGilTree:
    """
    Test results against hardcoded values Gil independently worked out
    """

    def test_gil_tree(self):
        for cache_inside in [False, True]:
            ts = utility_functions.gils_example_tree()
            span_data = SpansBySamples(ts)
            prior_distr = "lognorm"
            Ne = 0.5
            priors = ConditionalCoalescentTimes(None, Ne, prior_distr=prior_distr)
            priors.add(ts.num_samples, approximate=False)
            grid = np.array([0, 0.1, 0.2, 0.5, 1, 2, 5])
            mixture_prior = priors.get_mixture_prior_params(span_data)
            prior_vals = fill_priors(
                mixture_prior, grid, ts, 1, prior_distr=prior_distr
            )
            prior_vals.grid_data[0] = [0, 0.5, 0.3, 0.1, 0.05, 0.02, 0.03]
            prior_vals.grid_data[1] = [0, 0.05, 0.1, 0.2, 0.45, 0.1, 0.1]
            mut_rate = 1
            eps = 0.01
            lls = Likelihoods(ts, grid, mut_rate, eps=eps, normalize=False)
            lls.precalculate_mutation_likelihoods()
            algo = InOutAlgorithms(prior_vals, lls)
            algo.inside_pass(normalize=False, cache_inside=cache_inside)
            algo.outside_pass(normalize=False)
            assert np.allclose(
                np.sum(algo.inside.grid_data * algo.outside.grid_data, axis=1),
                [7.44449e-05, 7.44449e-05],
            )
            assert np.allclose(
                np.sum(algo.inside.grid_data * algo.outside.grid_data, axis=1),
                np.sum(algo.inside.grid_data[-1]),
            )


class TestOutsideEdgesOrdering:
    """
    Test that edges_by_child_desc() and edges_by_child_then_parent_desc() order edges
    correctly.
    """

    def edges_ordering(self, ts, fn):
        fixed_nodes = set(ts.samples())
        Ne = 1
        priors = tsdate.build_prior_grid(ts, Ne)
        mut_rate = None
        liklhd = LogLikelihoods(
            ts,
            priors.timepoints,
            mut_rate,
            eps=1e-6,
            fixed_node_set=fixed_nodes,
            progress=False,
        )
        dynamic_prog = InOutAlgorithms(priors, liklhd, progress=False)
        if fn == "outside_pass":
            edges_by_child = dynamic_prog.edges_by_child_desc()
            seen_children = list()
            last_child_time = None

            for child, edges in edges_by_child:
                for edge in edges:
                    assert edge.child not in seen_children
                cur_child_time = ts.tables.nodes.time[child]
                if last_child_time:
                    assert cur_child_time <= last_child_time
                seen_children.append(child)
                last_child_time = ts.tables.nodes.time[child]
        elif fn == "outside_maximization":
            edges_by_child = dynamic_prog.edges_by_child_then_parent_desc()
            seen_children = list()
            last_child_time = None

            for child, edges in edges_by_child:
                last_parent_time = None
                for edge in edges:
                    cur_parent_time = ts.tables.nodes.time[edge.parent]
                    if last_parent_time:
                        assert cur_parent_time >= last_parent_time
                    last_parent_time = cur_parent_time
                assert child not in seen_children
                cur_child_time = ts.tables.nodes.time[child]
                if last_child_time:
                    assert cur_child_time <= last_child_time

                seen_children.append(child)
                last_child_time = ts.tables.nodes.time[child]

    def test_two_tree_outside_traversal(self):
        """
        This is for the outside algorithm, where we simply want to traverse the ts
        from oldest child nodes to youngest, grouping all child nodes of same id
        together. In the outside maximization algorithm, we want to traverse the ts from
        oldest child nodes to youngest, grouping all child nodes of same id together.
        """
        ts = utility_functions.two_tree_two_mrcas()
        self.edges_ordering(ts, "outside_pass")
        self.edges_ordering(ts, "outside_maximization")

    def test_simulated_inferred_outside_traversal(self):
        ts = msprime.simulate(
            500,
            Ne=10000,
            length=5e4,
            mutation_rate=1e-8,
            recombination_rate=1e-8,
            random_seed=12,
        )
        sample_data = tsinfer.SampleData.from_tree_sequence(ts, use_sites_time=False)
        inferred_ts = tsinfer.infer(sample_data).simplify()
        self.edges_ordering(inferred_ts, "outside_pass")
        self.edges_ordering(inferred_ts, "outside_maximization")


class TestMaximization:
    """
    Test the outside maximization function
    """

    def run_outside_maximization(self, ts, prior_distr="lognorm"):
        Ne = 0.5
        priors = tsdate.build_prior_grid(ts, Ne, prior_distribution=prior_distr)
        mut_rate = 1
        eps = 1e-6
        lls = Likelihoods(ts, priors.timepoints, mut_rate, eps=eps)
        lls.precalculate_mutation_likelihoods()
        algo = InOutAlgorithms(priors, lls)
        algo.inside_pass()
        return lls, algo, algo.outside_maximization(eps=eps)

    def test_one_tree_n2(self):
        ts = utility_functions.single_tree_ts_n2()
        for prior_distr in ("lognorm", "gamma"):
            lls, algo, maximized_ages = self.run_outside_maximization(ts, prior_distr)
            assert np.array_equal(
                maximized_ages,
                np.array([0, 0, lls.timepoints[np.argmax(algo.inside[2])]]),
            )

    def test_one_tree_n3(self):
        ts = utility_functions.single_tree_ts_n3()
        for prior_distr in ("lognorm", "gamma"):
            lls, algo, maximized_ages = self.run_outside_maximization(ts, prior_distr)
            node_4 = lls.timepoints[np.argmax(algo.inside[4])]
            ll_mut = scipy.stats.poisson.pmf(
                0,
                (node_4 - lls.timepoints[: np.argmax(algo.inside[4]) + 1] + 1e-6)
                * 1
                * 1,
            )
            result = ll_mut / np.max(ll_mut)
            inside_val = algo.inside[3][: (np.argmax(algo.inside[4]) + 1)]
            node_3 = lls.timepoints[
                np.argmax(result[: np.argmax(algo.inside[4]) + 1] * inside_val)
            ]
            assert np.array_equal(maximized_ages, np.array([0, 0, 0, node_3, node_4]))

    def test_two_tree_ts(self):
        ts = utility_functions.two_tree_ts()
        for prior_distr in ("lognorm", "gamma"):
            lls, algo, maximized_ages = self.run_outside_maximization(ts, prior_distr)
            node_5 = lls.timepoints[np.argmax(algo.inside[5])]
            ll_mut = scipy.stats.poisson.pmf(
                0,
                (node_5 - lls.timepoints[: np.argmax(algo.inside[5]) + 1] + 1e-6)
                * 1
                * 0.8,
            )
            result = ll_mut / np.max(ll_mut)
            inside_val = algo.inside[4][: (np.argmax(algo.inside[5]) + 1)]
            node_4 = lls.timepoints[
                np.argmax(result[: np.argmax(algo.inside[5]) + 1] * inside_val)
            ]
            ll_mut = scipy.stats.poisson.pmf(
                0,
                (node_4 - lls.timepoints[: np.argmax(algo.inside[4]) + 1] + 1e-6)
                * 1
                * 0.2,
            )
            result = ll_mut / np.max(ll_mut)
            inside_val = algo.inside[3][: (np.argmax(algo.inside[4]) + 1)]
            node_3 = lls.timepoints[
                np.argmax(result[: np.argmax(algo.inside[4]) + 1] * inside_val)
            ]
            assert np.array_equal(
                maximized_ages, np.array([0, 0, 0, node_3, node_4, node_5])
            )


class TestDate:
    """
    Test inputs to tsdate.date()
    """

    def test_date_input(self):
        ts = utility_functions.single_tree_ts_n2()
        with pytest.raises(ValueError):
            tsdate.date(ts, mutation_rate=None, Ne=1, method="foobar")

    def test_sample_as_parent_fails(self):
        ts = utility_functions.single_tree_ts_n3_sample_as_parent()
        with pytest.raises(NotImplementedError):
            tsdate.date(ts, mutation_rate=None, Ne=1)

    def test_recombination_not_implemented(self):
        ts = utility_functions.single_tree_ts_n2()
        with pytest.raises(NotImplementedError):
            tsdate.date(ts, mutation_rate=None, Ne=1, recombination_rate=1e-8)

    def test_Ne_and_priors(self):
        ts = utility_functions.single_tree_ts_n2()
        with pytest.raises(ValueError):
            priors = tsdate.build_prior_grid(ts, Ne=1)
            tsdate.date(ts, mutation_rate=None, Ne=1, priors=priors)

    def test_no_Ne_priors(self):
        ts = utility_functions.single_tree_ts_n2()
        with pytest.raises(ValueError):
            tsdate.date(ts, mutation_rate=None, Ne=None, priors=None)


class TestBuildPriorGrid:
    """
    Test tsdate.build_prior_grid() works as expected
    """

    def test_bad_timepoints(self):
        ts = msprime.simulate(2, random_seed=123)
        Ne = 1
        for bad in [
            -1,
            np.array([1]),
            np.array([-1, 2, 3]),
            np.array([1, 1, 1]),
            "foobar",
        ]:
            with pytest.raises(ValueError):
                tsdate.build_prior_grid(ts, Ne, timepoints=bad)
        for bad in [np.array(["hello", "there"])]:
            with pytest.raises(TypeError):
                tsdate.build_prior_grid(ts, Ne, timepoints=bad)

    def test_bad_prior_distr(self):
        ts = msprime.simulate(2, random_seed=12)
        Ne = 1
        with pytest.raises(ValueError):
            tsdate.build_prior_grid(ts, Ne, prior_distribution="foobar")

    def test_bad_Ne(self):
        ts = msprime.simulate(2, random_seed=12)
        with pytest.raises(ValueError):
            tsdate.build_prior_grid(ts, Ne=-10)


class TestPosteriorMeanVar:
    """
    Test posterior_mean_var works as expected
    """

    def test_posterior_mean_var(self):
        ts = utility_functions.single_tree_ts_n2()
        grid = np.array([0, 1.2, 2])
        for distr in ("gamma", "lognorm"):
            posterior, algo = TestTotalFunctionalValueTree().find_posterior(ts, distr)
            ts_node_metadata, mn_post, vr_post = posterior_mean_var(ts, grid, posterior)
            assert np.array_equal(
                mn_post, [0, 0, np.sum(grid * posterior[2]) / np.sum(posterior[2])]
            )

    def test_node_metadata_single_tree_n2(self):
        ts = utility_functions.single_tree_ts_n2()
        grid = np.array([0, 1.2, 2])
        posterior, algo = TestTotalFunctionalValueTree().find_posterior(ts, "lognorm")
        ts_node_metadata, mn_post, vr_post = posterior_mean_var(ts, grid, posterior)
        assert json.loads(ts_node_metadata.node(2).metadata)["mn"] == mn_post[2]
        assert json.loads(ts_node_metadata.node(2).metadata)["vr"] == vr_post[2]

    def test_node_metadata_simulated_tree(self):
        larger_ts = msprime.simulate(
            10, mutation_rate=1, recombination_rate=1, length=20, random_seed=12
        )
        _, mn_post, _, _, eps, _ = get_dates(larger_ts, mutation_rate=None, Ne=10000)
        dated_ts = date(larger_ts, Ne=10000, mutation_rate=None)
        metadata = dated_ts.tables.nodes.metadata
        metadata_offset = dated_ts.tables.nodes.metadata_offset
        unconstrained_mn = [
            json.loads(met.decode())["mn"]
            for met in tskit.unpack_bytes(metadata, metadata_offset)
            if len(met.decode()) > 0
        ]
        assert np.array_equal(unconstrained_mn, mn_post[larger_ts.num_samples :])
        assert np.all(
            dated_ts.tables.nodes.time[larger_ts.num_samples :]
            >= mn_post[larger_ts.num_samples :]
        )


class TestConstrainAgesTopo:
    """
    Test constrain_ages_topo works as expected
    """

    def test_constrain_ages_topo(self):
        """
        Set node 3 to be older than node 4 in two_tree_ts
        """
        ts = utility_functions.two_tree_ts()
        post_mn = np.array([0.0, 0.0, 0.0, 2.0, 1.0, 3.0])
        eps = 1e-6
        nodes_to_date = np.array([3, 4, 5])
        constrained_ages = constrain_ages_topo(ts, post_mn, eps, nodes_to_date)
        assert np.array_equal(
            np.array([0.0, 0.0, 0.0, 2.0, 2.000001, 3.0]), constrained_ages
        )

    def test_constrain_ages_topo_no_nodes_to_date(self):
        ts = utility_functions.two_tree_ts()
        post_mn = np.array([0.0, 0.0, 0.0, 2.0, 1.0, 3.0])
        eps = 1e-6
        nodes_to_date = None
        constrained_ages = constrain_ages_topo(ts, post_mn, eps, nodes_to_date)
        assert np.array_equal(
            np.array([0.0, 0.0, 0.0, 2.0, 2.000001, 3.0]), constrained_ages
        )

    def test_constrain_ages_topo_unary_nodes_unordered(self):
        ts = utility_functions.single_tree_ts_with_unary()
        post_mn = np.array([0.0, 0.0, 0.0, 2.0, 1.0, 0.5, 5.0, 1.0])
        eps = 1e-6
        constrained_ages = constrain_ages_topo(ts, post_mn, eps)
        assert np.allclose(
            np.array([0.0, 0.0, 0.0, 2.0, 2.000001, 2.000002, 5.0, 5.000001]),
            constrained_ages,
        )

    def test_constrain_ages_topo_part_dangling(self):
        ts = utility_functions.two_tree_ts_n2_part_dangling()
        post_mn = np.array([1.0, 0.0, 0.0, 0.1, 0.05])
        eps = 1e-6
        constrained_ages = constrain_ages_topo(ts, post_mn, eps)
        assert np.allclose(
            np.array([1.0, 0.0, 0.0, 1.000001, 1.000002]), constrained_ages
        )

    def test_constrain_ages_topo_sample_as_parent(self):
        ts = utility_functions.single_tree_ts_n3_sample_as_parent()
        post_mn = np.array([0.0, 0.0, 0.0, 3.0, 1.0])
        eps = 1e-6
        constrained_ages = constrain_ages_topo(ts, post_mn, eps)
        assert np.allclose(np.array([0.0, 0.0, 0.0, 3.0, 3.000001]), constrained_ages)

    def test_two_tree_ts_n3_non_contemporaneous(self):
        ts = utility_functions.two_tree_ts_n3_non_contemporaneous()
        post_mn = np.array([0.0, 0.0, 3.0, 4.0, 0.1, 4.1])
        eps = 1e-6
        constrained_ages = constrain_ages_topo(ts, post_mn, eps)
        assert np.allclose(
            np.array([0.0, 0.0, 3.0, 4.0, 4.000001, 4.1]), constrained_ages
        )


class TestPreprocessTs(unittest.TestCase):
    """
    Test preprocess_ts works as expected
    """

    def verify(self, ts, minimum_gap=None, remove_telomeres=None, **kwargs):
        with self.assertLogs("tsdate.util", level="INFO") as logs:
            if minimum_gap is not None and remove_telomeres is not None:
                ts = tsdate.preprocess_ts(
                    ts, minimum_gap=minimum_gap, remove_telomeres=remove_telomeres
                )
            elif minimum_gap is not None and remove_telomeres is None:
                ts = tsdate.preprocess_ts(ts, minimum_gap=minimum_gap)
            elif remove_telomeres is not None and minimum_gap is None:
                ts = tsdate.preprocess_ts(ts, remove_telomeres=remove_telomeres)
            else:
                ts = tsdate.preprocess_ts(ts, **kwargs)
        messages = [record.msg for record in logs.records]
        assert "Beginning preprocessing" in messages
        return ts

    def test_no_sites(self):
        ts = utility_functions.two_tree_ts()
        with pytest.raises(ValueError):
            tsdate.preprocess_ts(ts)

    def test_invariant_sites(self):
        # Test that passing kwargs to simplify works as expected
        ts = utility_functions.site_no_mutations()
        with warnings.catch_warnings(record=True) as w:
            removed = self.verify(ts)
            assert removed.num_sites == 0
            assert len(w) == 1
        assert (
            tsdate.preprocess_ts(ts, **{"filter_sites": False}).num_sites
            == ts.num_sites
        )

    def test_no_intervals(self):
        ts = utility_functions.two_tree_mutation_ts()
        assert ts.tables.edges == self.verify(ts, remove_telomeres=False).tables.edges
        assert ts.tables.edges == self.verify(ts, minimum_gap=0.05).tables.edges

    def test_delete_interval(self):
        ts = utility_functions.ts_w_data_desert(40, 60, 100)
        trimmed = self.verify(ts, minimum_gap=20, remove_telomeres=False)
        lefts = trimmed.tables.edges.left
        rights = trimmed.tables.edges.right
        assert not np.any(np.logical_and(lefts > 41, lefts < 59))
        assert not np.any(np.logical_and(rights > 41, rights < 59))

    def test_remove_telomeres(self):
        ts = utility_functions.ts_w_data_desert(0, 5, 100)
        removed = self.verify(ts, minimum_gap=ts.get_sequence_length())
        lefts = removed.tables.edges.left
        rights = removed.tables.edges.right
        assert not np.any(np.logical_and(lefts > 0, lefts < 4))
        assert not np.any(np.logical_and(rights > 0, rights < 4))
        ts = utility_functions.ts_w_data_desert(95, 100, 100)
        removed = self.verify(ts, minimum_gap=ts.get_sequence_length())
        lefts = removed.tables.edges.left
        rights = removed.tables.edges.right
        assert not np.any(np.logical_and(lefts > 96, lefts < 100))
        assert not np.any(np.logical_and(rights > 96, rights < 100))


class TestNodeTimes:
    """
    Test node_times works as expected.
    """

    def test_node_times(self):
        larger_ts = msprime.simulate(
            10, mutation_rate=1, recombination_rate=1, length=20, random_seed=12
        )
        dated = date(larger_ts, mutation_rate=None, Ne=10000)
        node_ages = nodes_time_unconstrained(dated)
        assert np.all(dated.tables.nodes.time[:] >= node_ages)

    def test_fails_unconstrained(self):
        ts = utility_functions.two_tree_mutation_ts()
        with pytest.raises(ValueError):
            nodes_time_unconstrained(ts)


class TestSiteTimes:
    """
    Test sites_time works as expected
    """

    def test_no_sites(self):
        ts = utility_functions.two_tree_ts()
        with pytest.raises(ValueError):
            tsdate.sites_time_from_ts(ts)

    def test_node_selection_param(self):
        ts = utility_functions.two_tree_mutation_ts()
        with pytest.raises(ValueError):
            tsdate.sites_time_from_ts(ts, node_selection="sibling")

    def test_sites_time_insideoutside(self):
        ts = utility_functions.two_tree_mutation_ts()
        dated = tsdate.date(ts, mutation_rate=None, Ne=1)
        _, mn_post, _, _, eps, _ = get_dates(ts, mutation_rate=None, Ne=1)
        assert np.array_equal(
            mn_post[ts.tables.mutations.node],
            tsdate.sites_time_from_ts(dated, unconstrained=True, min_time=0),
        )
        assert np.array_equal(
            dated.tables.nodes.time[ts.tables.mutations.node],
            tsdate.sites_time_from_ts(dated, unconstrained=False, min_time=0),
        )

    def test_sites_time_maximization(self):
        ts = utility_functions.two_tree_mutation_ts()
        dated = tsdate.date(ts, Ne=1, mutation_rate=1, method="maximization")
        assert np.array_equal(
            dated.tables.nodes.time[ts.tables.mutations.node],
            tsdate.sites_time_from_ts(dated, unconstrained=False, min_time=0),
        )

    def test_sites_time_node_selection(self):
        ts = utility_functions.two_tree_mutation_ts()
        dated = tsdate.date(ts, Ne=1, mutation_rate=1)
        sites_time_child = tsdate.sites_time_from_ts(
            dated, node_selection="child", min_time=0
        )
        dated_nodes_time = nodes_time_unconstrained(dated)
        assert np.array_equal(
            dated_nodes_time[ts.tables.mutations.node], sites_time_child
        )

        sites_time_parent = tsdate.sites_time_from_ts(
            dated, node_selection="parent", min_time=0
        )
        parent_sites_check = np.zeros(dated.num_sites)
        for tree in dated.trees():
            for site in tree.sites():
                for mut in site.mutations:
                    parent_sites_check[site.id] = dated_nodes_time[
                        tree.parent(mut.node)
                    ]
        assert np.array_equal(parent_sites_check, sites_time_parent)

        sites_time_arithmetic = tsdate.sites_time_from_ts(
            dated, node_selection="arithmetic", min_time=0
        )
        arithmetic_sites_check = np.zeros(dated.num_sites)
        for tree in dated.trees():
            for site in tree.sites():
                for mut in site.mutations:
                    arithmetic_sites_check[site.id] = (
                        dated_nodes_time[mut.node]
                        + dated_nodes_time[tree.parent(mut.node)]
                    ) / 2
        assert np.array_equal(arithmetic_sites_check, sites_time_arithmetic)

        sites_time_geometric = tsdate.sites_time_from_ts(
            dated, node_selection="geometric", min_time=0
        )
        geometric_sites_check = np.zeros(dated.num_sites)
        for tree in dated.trees():
            for site in tree.sites():
                for mut in site.mutations:
                    geometric_sites_check[site.id] = np.sqrt(
                        dated_nodes_time[mut.node]
                        * dated_nodes_time[tree.parent(mut.node)]
                    )
        assert np.array_equal(geometric_sites_check, sites_time_geometric)

    def test_sites_time_singletons(self):
        """Singletons should be allocated min_time"""
        ts = utility_functions.single_tree_ts_2mutations_singletons_n3()
        sites_time = tsdate.sites_time_from_ts(ts, unconstrained=False, min_time=0)
        assert np.array_equal(sites_time, [0])
        sites_time = tsdate.sites_time_from_ts(ts, unconstrained=False)
        assert np.array_equal(sites_time, [1])

    def test_sites_time_nonvariable(self):
        ts = utility_functions.single_tree_ts_2mutations_singletons_n3()
        tables = ts.dump_tables()
        tables.mutations.clear()
        ts = tables.tree_sequence()
        sites_time = tsdate.sites_time_from_ts(ts, unconstrained=False)
        assert np.all(np.isnan(sites_time))

    def test_sites_time_root_mutation(self):
        ts = utility_functions.single_tree_ts_2mutations_singletons_n3()
        tables = ts.dump_tables()
        tables.mutations.clear()
        tables.mutations.add_row(site=0, derived_state="1", node=ts.first().root)
        ts = tables.tree_sequence()
        sites_time = tsdate.sites_time_from_ts(ts, unconstrained=False)
        assert sites_time[0] == ts.node(ts.first().root).time

    def test_sites_time_multiple_mutations(self):
        ts = utility_functions.single_tree_ts_2mutations_n3()
        sites_time = tsdate.sites_time_from_ts(ts, unconstrained=False)
        assert np.array_equal(sites_time, [10])

    def test_sites_time_simulated(self):
        larger_ts = msprime.simulate(
            10, mutation_rate=1, recombination_rate=1, length=20, random_seed=12
        )
        _, mn_post, _, _, _, _ = get_dates(larger_ts, mutation_rate=None, Ne=10000)
        dated = date(larger_ts, mutation_rate=None, Ne=10000)
        assert np.array_equal(
            mn_post[larger_ts.tables.mutations.node],
            tsdate.sites_time_from_ts(dated, unconstrained=True, min_time=0),
        )
        assert np.array_equal(
            dated.tables.nodes.time[larger_ts.tables.mutations.node],
            tsdate.sites_time_from_ts(dated, unconstrained=False, min_time=0),
        )


class TestSampleDataTimes:
    """
    Test add_sampledata_times
    """

    def test_wrong_number_of_sites(self):
        ts = utility_functions.single_tree_ts_2mutations_n3()
        sites_time = tsdate.sites_time_from_ts(ts, unconstrained=False)
        sites_time = np.append(sites_time, [10])
        samples = tsinfer.formats.SampleData.from_tree_sequence(
            ts, use_sites_time=False
        )
        with pytest.raises(ValueError):
            tsdate.add_sampledata_times(samples, sites_time)

    def test_historical_samples(self):
        samples = [msprime.Sample(population=0, time=0) for i in range(10)]
        ancients = [msprime.Sample(population=0, time=1000) for i in range(10)]
        samps = samples + ancients
        ts = msprime.simulate(
            samples=samps,
            mutation_rate=1e-8,
            recombination_rate=1e-8,
            Ne=10000,
            length=1e4,
            random_seed=12,
        )
        ancient_samples = np.where(ts.tables.nodes.time[:][ts.samples()] != 0)[
            0
        ].astype("int32")
        ancient_samples_times = ts.tables.nodes.time[ancient_samples]

        samples = tsinfer.formats.SampleData.from_tree_sequence(ts)
        inferred = tsinfer.infer(samples).simplify(filter_sites=False)
        dated = date(inferred, 10000, 1e-8)
        sites_time = tsdate.sites_time_from_ts(dated)
        # Add in the original individual times
        ind_dated_sd = samples.copy()
        ind_dated_sd.individuals_time[
            :
        ] = tsinfer.formats.SampleData.from_tree_sequence(
            ts, use_individuals_time=True, use_sites_time=True
        ).individuals_time[
            :
        ]
        ind_dated_sd.finalise()
        dated_samples = tsdate.add_sampledata_times(ind_dated_sd, sites_time)
        for variant in ts.variants(samples=ancient_samples):
            if np.any(variant.genotypes == 1):
                ancient_bound = np.max(ancient_samples_times[variant.genotypes == 1])
                assert dated_samples.sites_time[variant.site.id] >= ancient_bound

    def test_sampledata(self):
        samples = [msprime.Sample(population=0, time=0) for i in range(10)]
        ancients = [msprime.Sample(population=0, time=1000) for i in range(10)]
        samps = samples + ancients
        ts = msprime.simulate(
            samples=samps,
            mutation_rate=1e-8,
            recombination_rate=1e-8,
            Ne=10000,
            length=1e4,
            random_seed=12,
        )
        samples = tsinfer.formats.SampleData.from_tree_sequence(
            ts, use_sites_time=False
        )
        inferred = tsinfer.infer(samples).simplify()
        dated = date(inferred, 10000, 1e-8)
        sites_time = tsdate.sites_time_from_ts(dated)
        sites_bound = samples.min_site_times(individuals_only=True)
        check_sites_time = np.maximum(sites_time, sites_bound)
        copy = tsdate.add_sampledata_times(samples, sites_time)
        assert np.array_equal(copy.sites_time[:], check_sites_time)


class TestHistoricalExample:
    def historical_samples_example(self):
        samples = [
            msprime.Sample(population=0, time=0),
            msprime.Sample(0, 0),
            msprime.Sample(0, 0),
            msprime.Sample(0, 1.0),
        ]
        return msprime.simulate(
            samples=samples, mutation_rate=1, length=1e2, random_seed=12
        )

    def test_historical_samples(self):
        ts = self.historical_samples_example()
        modern_samples = tsinfer.SampleData.from_tree_sequence(
            ts.simplify(ts.samples(time=0), filter_sites=False)
        )
        inferred_ts = tsinfer.infer(modern_samples).simplify(filter_sites=False)
        dated_ts = tsdate.date(inferred_ts, Ne=1, mutation_rate=1)
        site_times = tsdate.sites_time_from_ts(dated_ts)
        # make a sd file with historical individual times
        samples = tsinfer.SampleData.from_tree_sequence(
            ts,
            use_individuals_time=True,
            # site times will be replaced, but True needed below to avoid timescale clash
            use_sites_time=True,
        )
        dated_samples = tsdate.add_sampledata_times(samples, site_times)
        ancestors = tsinfer.generate_ancestors(dated_samples)
        ancestors_w_proxy = ancestors.insert_proxy_samples(
            dated_samples, allow_mutation=True
        )
        ancestors_ts = tsinfer.match_ancestors(dated_samples, ancestors_w_proxy)
        reinferred_ts = tsinfer.match_samples(
            dated_samples, ancestors_ts, force_sample_times=True
        )
        assert reinferred_ts.num_samples == ts.num_samples
        assert reinferred_ts.num_sites == ts.num_sites
        assert np.array_equal(
            reinferred_ts.tables.nodes.time[reinferred_ts.samples()],
            ts.tables.nodes.time[ts.samples()],
        )
