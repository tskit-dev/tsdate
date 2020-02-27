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
import unittest
import collections

import numpy as np
import scipy
import tskit  # NOQA
import msprime
import tsinfer

import tsdate
from tsdate.date import (SpansBySamples, PriorParams,
                         ConditionalCoalescentTimes, fill_prior, Likelihoods,
                         LogLikelihoods, LogLikelihoodsStreaming, InOutAlgorithms,
                         NodeGridValues, gamma_approx, constrain_ages_topo)  # NOQA

from tests import utility_functions


class TestBasicFunctions(unittest.TestCase):
    """
    Test for some of the basic functions used in tsdate
    """

    def test_alpha_prob(self):
        self.assertEqual(ConditionalCoalescentTimes.m_prob(2, 2, 3), 1.)
        self.assertEqual(ConditionalCoalescentTimes.m_prob(2, 2, 4), 0.25)

    def test_tau_expect(self):
        self.assertEqual(ConditionalCoalescentTimes.tau_expect(10, 10), 1.8)
        self.assertEqual(ConditionalCoalescentTimes.tau_expect(10, 100), 0.09)
        self.assertEqual(ConditionalCoalescentTimes.tau_expect(100, 100), 1.98)
        self.assertEqual(ConditionalCoalescentTimes.tau_expect(5, 10), 0.4)

    def test_tau_squared_conditional(self):
        self.assertAlmostEqual(
            ConditionalCoalescentTimes.tau_squared_conditional(1, 10), 4.3981418)
        self.assertAlmostEqual(
            ConditionalCoalescentTimes.tau_squared_conditional(100, 100),
            -4.87890977e-18)

    def test_tau_var(self):
        self.assertEqual(
            ConditionalCoalescentTimes.tau_var(2, 2), 1)
        self.assertAlmostEqual(
            ConditionalCoalescentTimes.tau_var(10, 20), 0.0922995960)
        self.assertAlmostEqual(
            ConditionalCoalescentTimes.tau_var(50, 50), 1.15946186)

    def test_gamma_approx(self):
        self.assertEqual(gamma_approx(2, 1), (4., 2.))
        self.assertEqual(gamma_approx(0.5, 0.1), (2.5, 5.0))


class TestNodeTipWeights(unittest.TestCase):
    def verify_weights(self, ts):
        span_data = SpansBySamples(ts)
        # Check all non-sample nodes in a tree are represented
        nonsample_nodes = collections.defaultdict(float)
        for tree in ts.trees():
            for n in tree.nodes():
                if not tree.is_sample(n):
                    # do not count a span of a node where there are no sample descendants
                    if tree.num_samples(n) > 0:
                        nonsample_nodes[n] += tree.span
        self.assertEqual(set(span_data.nodes_to_date), set(nonsample_nodes.keys()))
        for id, span in nonsample_nodes.items():
            self.assertAlmostEqual(span, span_data.node_spans[id])
        for focal_node in span_data.nodes_to_date:
            wt = 0
            for num_samples, weights in span_data.get_weights(focal_node).items():
                self.assertTrue(0 <= focal_node < ts.num_nodes)
                wt += np.sum(weights['weight'])
                self.assertLessEqual(max(weights['descendant_tips']), ts.num_samples)
            self.assertAlmostEqual(wt, 1.0)
        return span_data

    def test_one_tree_n2(self):
        ts = utility_functions.single_tree_ts_n2()
        span_data = self.verify_weights(ts)
        # with a single tree there should only be one weight
        for node in span_data.nodes_to_date:
            self.assertTrue(len(span_data.get_weights(node)), 1)
        self.assertTrue(2 in span_data.get_weights(2)[ts.num_samples]['descendant_tips'])

    def test_one_tree_n3(self):
        ts = utility_functions.single_tree_ts_n3()
        n = ts.num_samples
        span_data = self.verify_weights(ts)
        # with a single tree there should only be one weight
        for node in span_data.nodes_to_date:
            self.assertTrue(len(span_data.get_weights(node)), 1)
        for nd, expd_tips in [
                (4, 3),   # Node 4 (root) expected to have 3 descendant tips
                (3, 2)]:  # Node 3 (1st internal node) expected to have 2 descendant tips
            self.assertTrue(
                np.isin(span_data.get_weights(nd)[n]['descendant_tips'], expd_tips))

    def test_one_tree_n4(self):
        ts = utility_functions.single_tree_ts_n4()
        n = ts.num_samples
        span_data = self.verify_weights(ts)
        # with a single tree there should only be one weight
        for node in span_data.nodes_to_date:
            self.assertTrue(len(span_data.get_weights(node)), 1)
        for nd, expd_tips in [
                (6, 4),   # Node 6 (root) expected to have 4 descendant tips
                (5, 3),   # Node 5 (1st internal node) expected to have 3 descendant tips
                (4, 2)]:  # Node 4 (2nd internal node) expected to have 3 descendant tips
            self.assertTrue(
                np.isin(span_data.get_weights(nd)[n]['descendant_tips'], expd_tips))

    def test_two_trees(self):
        ts = utility_functions.two_tree_ts()
        n = ts.num_samples
        span_data = self.verify_weights(ts)
        self.assertEqual(span_data.lookup_weight(5, n, 3), 1.0)  # Root on R tree
        self.assertEqual(span_data.lookup_weight(4, n, 3), 0.2)  # Root on L tree ...
        # ... but internal node on R tree
        self.assertEqual(span_data.lookup_weight(4, n, 2), 0.8)
        self.assertEqual(span_data.lookup_weight(3, n, 2), 1.0)  # Internal nd on L tree

    def test_missing_tree(self):
        ts = utility_functions.two_tree_ts().keep_intervals(
            [(0, 0.2)], simplify=False)
        n = ts.num_samples
        # Here we have no reference in the trees to node 6
        self.assertRaises(ValueError, SpansBySamples, ts)
        ts = ts.simplify()
        span_data = self.verify_weights(ts)
        # Root on (deleted) R tree is missing
        self.assertTrue(5 not in span_data.nodes_to_date)
        self.assertEqual(span_data.lookup_weight(4, n, 3), 1.0)  # Root on L tree ...
        # ... but internal on (deleted) R tree
        self.assertFalse(np.isin(span_data.get_weights(4)[n]['descendant_tips'], 2))
        self.assertEqual(span_data.lookup_weight(3, n, 2), 1.0)  # Internal nd on L tree

    def test_tree_with_unary_nodes(self):
        ts = utility_functions.single_tree_ts_with_unary()
        n = ts.num_samples
        span_data = self.verify_weights(ts)
        self.assertEqual(span_data.lookup_weight(7, n, 3), 1.0)
        self.assertEqual(span_data.lookup_weight(6, n, 1), 0.5)
        self.assertEqual(span_data.lookup_weight(6, n, 3), 0.5)
        self.assertEqual(span_data.lookup_weight(5, n, 2), 0.5)
        self.assertEqual(span_data.lookup_weight(5, n, 3), 0.5)
        self.assertEqual(span_data.lookup_weight(4, n, 2), 0.75)
        self.assertEqual(span_data.lookup_weight(4, n, 3), 0.25)
        self.assertEqual(span_data.lookup_weight(3, n, 2), 1.0)

    def test_polytomy_tree(self):
        ts = utility_functions.polytomy_tree_ts()
        span_data = self.verify_weights(ts)
        self.assertEqual(span_data.lookup_weight(3, ts.num_samples, 3), 1.0)

    def test_larger_find_node_tip_weights(self):
        ts = msprime.simulate(10, recombination_rate=5,
                              mutation_rate=5, random_seed=123)
        self.assertGreater(ts.num_trees, 1)
        self.verify_weights(ts)

    def test_dangling_nodes_fail(self):
        ts = utility_functions.single_tree_ts_n3()
        # Mark node 0 as a non-sample node, which should make it dangling
        tables = ts.dump_tables()
        flags = tables.nodes.flags
        flags[0] = flags[0] & (~tskit.NODE_IS_SAMPLE)
        tables.nodes.flags = flags
        self.assertRaises(ValueError, self.verify_weights, tables.tree_sequence())

    @unittest.skip("YAN to fix")
    def test_truncated_nodes(self):
        Ne = 1e2
        ts = msprime.simulate(
            10, Ne=Ne, length=400, recombination_rate=1e-4, random_seed=12)
        truncated_ts = utility_functions.truncate_ts_samples(
            ts, average_span=200, random_seed=123)
        span_data = self.verify_weights(truncated_ts)
        raise NotImplementedError(str(span_data))


class TestMakePrior(unittest.TestCase):
    # We only test make_prior() on single trees
    def verify_prior(self, ts, prior_distr):
        # Check prior contains all possible tips
        priors = ConditionalCoalescentTimes(None, prior_distr=prior_distr)
        priors.add(ts.num_samples)
        prior_df = priors[ts.num_samples]
        self.assertEqual(prior_df.shape[0], ts.num_samples + 1)
        return(prior_df)

    def test_one_tree_n2(self):
        ts = utility_functions.single_tree_ts_n2()
        prior = self.verify_prior(ts, 'gamma')
        self.assertTrue(np.allclose(
            prior[2], PriorParams(alpha=1., beta=1., mean=1., var=1.)))
        prior = self.verify_prior(ts, 'lognorm')
        self.assertTrue(np.allclose(
            prior[2], PriorParams(alpha=-0.34657359, beta=0.69314718, mean=1., var=1.)))

    def test_one_tree_n3(self):
        ts = utility_functions.single_tree_ts_n3()
        prior2mv = {'mean': 1/3, 'var': 1/9}
        prior3mv = {'mean': 1+1/3, 'var': 1+1/9}
        prior = self.verify_prior(ts, 'lognorm')
        self.assertTrue(np.allclose(
            prior[2], PriorParams(alpha=-1.44518588, beta=0.69314718, **prior2mv)))
        self.assertTrue(np.allclose(
            prior[3], PriorParams(alpha=0.04492816, beta=0.48550782, **prior3mv)))
        prior = self.verify_prior(ts, 'gamma')
        self.assertTrue(np.allclose(
            prior[2], PriorParams(alpha=1., beta=3., **prior2mv)))
        self.assertTrue(np.allclose(
            prior[3], PriorParams(alpha=1.6, beta=1.2, **prior3mv)))

    def test_one_tree_n4(self):
        ts = utility_functions.single_tree_ts_n4()
        self.skipTest("Fill in values instead of np.nan")
        prior2mv = {'mean': np.nan, 'var': np.nan}
        prior3mv = {'mean': np.nan, 'var': np.nan}
        prior4mv = {'mean': np.nan, 'var': np.nan}

        prior = self.verify_prior(ts, 'lognorm')
        self.assertTrue(np.allclose(
            prior[2], PriorParams(alpha=np.nan, beta=np.nan, **prior2mv)))
        self.assertTrue(np.allclose(
            prior[3], PriorParams(alpha=np.nan, beta=np.nan, **prior3mv)))
        self.assertTrue(np.allclose(
            prior[4], PriorParams(alpha=np.nan, beta=np.nan, **prior4mv)))

        prior = self.verify_prior(ts, 'gamma')
        self.assertTrue(np.allclose(
            prior[2], PriorParams(alpha=np.nan, beta=np.nan, **prior2mv)))
        self.assertTrue(np.allclose(
            prior[3], PriorParams(alpha=np.nan, beta=np.nan, **prior3mv)))
        self.assertTrue(np.allclose(
            prior[4], PriorParams(alpha=np.nan, beta=np.nan, **prior4mv)))

    def test_polytomy_tree(self):
        ts = utility_functions.polytomy_tree_ts()
        self.skipTest("Fill in values instead of np.nan")
        prior3mv = {'mean': np.nan, 'var': np.nan}

        prior = self.verify_prior(ts, 'lognorm')
        self.assertTrue(np.allclose(
            prior[3], PriorParams(alpha=np.nan, beta=np.nan, **prior3mv)))

        prior = self.verify_prior(ts, 'gamma')
        self.assertTrue(np.allclose(
            prior[3], PriorParams(alpha=np.nan, beta=np.nan, **prior3mv)))

    def test_two_tree_ts(self):
        ts = utility_functions.two_tree_ts()
        self.skipTest("Fill in values instead of np.nan")
        prior2mv = {'mean': np.nan, 'var': np.nan}
        prior3mv = {'mean': np.nan, 'var': np.nan}

        prior = self.verify_prior(ts, 'lognorm')
        self.assertTrue(np.allclose(
            prior[2], PriorParams(alpha=np.nan, beta=np.nan, **prior2mv)))
        self.assertTrue(np.allclose(
            prior[3], PriorParams(alpha=np.nan, beta=np.nan, **prior3mv)))

        prior = self.verify_prior(ts, 'gamma')
        self.assertTrue(np.allclose(
            prior[2], PriorParams(alpha=np.nan, beta=np.nan, **prior2mv)))
        self.assertTrue(np.allclose(
            prior[3], PriorParams(alpha=np.nan, beta=np.nan, **prior3mv)))

    def test_single_tree_ts_with_unary(self):
        ts = utility_functions.single_tree_ts_with_unary()
        self.skipTest("Fill in values instead of np.nan")
        prior2mv = {'mean': np.nan, 'var': np.nan}
        prior3mv = {'mean': np.nan, 'var': np.nan}

        prior = self.verify_prior(ts, 'lognorm')
        self.assertTrue(np.allclose(
            prior[2], PriorParams(alpha=np.nan, beta=np.nan, **prior2mv)))
        self.assertTrue(np.allclose(
            prior[3], PriorParams(alpha=np.nan, beta=np.nan, **prior3mv)))

        prior = self.verify_prior(ts, 'gamma')
        self.assertTrue(np.allclose(
            prior[2], PriorParams(alpha=1., beta=3., **prior2mv)))
        self.assertTrue(np.allclose(
            prior[3], PriorParams(alpha=1.6, beta=1.2, **prior3mv)))

    def test_two_tree_mutation_ts(self):
        ts = utility_functions.two_tree_mutation_ts()
        self.skipTest("Fill in values instead of np.nan")
        prior2mv = {'mean': np.nan, 'var': np.nan}
        prior3mv = {'mean': np.nan, 'var': np.nan}

        prior = self.verify_prior(ts, 'lognorm')
        self.assertTrue(np.allclose(
            prior[2], PriorParams(alpha=np.nan, beta=np.nan, **prior2mv)))
        self.assertTrue(np.allclose(
            prior[3], PriorParams(alpha=np.nan, beta=np.nan, **prior3mv)))

        prior = self.verify_prior(ts, 'gamma')
        self.assertTrue(np.allclose(
            prior[2], PriorParams(alpha=1., beta=3., **prior2mv)))
        self.assertTrue(np.allclose(
            prior[3], PriorParams(alpha=1.6, beta=1.2, **prior3mv)))


class TestMixturePrior(unittest.TestCase):
    alpha_beta = [PriorParams.field_index('alpha'), PriorParams.field_index('beta')]

    def get_mixture_prior_params(self, ts, prior_distr):
        span_data = SpansBySamples(ts)
        priors = ConditionalCoalescentTimes(None, prior_distr=prior_distr)
        priors.add(ts.num_samples, approximate=False)
        mixture_prior = priors.get_mixture_prior_params(span_data)
        return(mixture_prior)

    def test_one_tree_n2(self):
        ts = utility_functions.single_tree_ts_n2()
        mixture_prior = self.get_mixture_prior_params(ts, 'gamma')
        self.assertTrue(
            np.allclose(mixture_prior[2, self.alpha_beta], [1., 1.]))
        mixture_prior = self.get_mixture_prior_params(ts, 'lognorm')
        self.assertTrue(
            np.allclose(mixture_prior[2,  self.alpha_beta], [-0.34657359, 0.69314718]))

    def test_one_tree_n3(self):
        ts = utility_functions.single_tree_ts_n3()
        mixture_prior = self.get_mixture_prior_params(ts, 'gamma')
        self.assertTrue(
            np.allclose(mixture_prior[3, self.alpha_beta], [1., 3.]))
        self.assertTrue(
            np.allclose(mixture_prior[4, self.alpha_beta], [1.6, 1.2]))
        mixture_prior = self.get_mixture_prior_params(ts, 'lognorm')
        self.assertTrue(
            np.allclose(mixture_prior[3, self.alpha_beta], [-1.44518588, 0.69314718]))
        self.assertTrue(
            np.allclose(mixture_prior[4, self.alpha_beta], [0.04492816, 0.48550782]))

    def test_one_tree_n4(self):
        ts = utility_functions.single_tree_ts_n4()
        mixture_prior = self.get_mixture_prior_params(ts, 'gamma')
        self.assertTrue(
            np.allclose(mixture_prior[4, self.alpha_beta], [0.81818182, 3.27272727]))
        self.assertTrue(
            np.allclose(mixture_prior[5, self.alpha_beta], [1.8, 3.6]))
        self.assertTrue(
            np.allclose(mixture_prior[6, self.alpha_beta], [1.97560976, 1.31707317]))

    def test_polytomy_tree(self):
        ts = utility_functions.polytomy_tree_ts()
        mixture_prior = self.get_mixture_prior_params(ts, 'gamma')
        self.assertTrue(
            np.allclose(mixture_prior[3, self.alpha_beta], [1.6, 1.2]))

    def test_two_trees(self):
        ts = utility_functions.two_tree_ts()
        mixture_prior = self.get_mixture_prior_params(ts, 'gamma')
        self.assertTrue(
            np.allclose(mixture_prior[3, self.alpha_beta], [1., 3.]))
        # Node 4 should be a mixture between 2 and 3 tips
        self.assertTrue(
            np.allclose(mixture_prior[4, self.alpha_beta], [0.60377, 1.13207]))
        self.assertTrue(
            np.allclose(mixture_prior[5, self.alpha_beta], [1.6, 1.2]))

    def test_single_tree_ts_with_unary(self):
        ts = utility_functions.single_tree_ts_with_unary()
        mixture_prior = self.get_mixture_prior_params(ts, 'gamma')
        # Root is a 3 tip prior
        self.assertTrue(
            np.allclose(mixture_prior[7, self.alpha_beta], [1.6, 1.2]))
        # Node 6 should be a 50:50 mixture between 1 and 3 tips
        self.assertTrue(
            np.allclose(mixture_prior[6, self.alpha_beta], [0.44444, 0.66666]))
        # Node 5 should be a 50:50 mixture of 2 and 3 tips
        self.assertTrue(
            np.allclose(mixture_prior[5, self.alpha_beta], [0.80645, 0.96774]))
        # Node 4 should be a 75:25 mixture of 2 and 3 tips
        self.assertTrue(
            np.allclose(mixture_prior[4, self.alpha_beta], [0.62025, 1.06329]))
        # Node 3 is a 2 tip prior
        self.assertTrue(
            np.allclose(mixture_prior[3, self.alpha_beta], [1., 3.]))

    def test_two_tree_mutation_ts(self):
        ts = utility_functions.two_tree_mutation_ts()
        mixture_prior = self.get_mixture_prior_params(ts, 'gamma')
        self.assertTrue(
            np.allclose(mixture_prior[3, self.alpha_beta], [1., 3.]))
        # Node 4 should be a mixture between 2 and 3 tips
        self.assertTrue(
            np.allclose(mixture_prior[4, self.alpha_beta], [0.60377, 1.13207]))
        self.assertTrue(
            np.allclose(mixture_prior[5, self.alpha_beta], [1.6, 1.2]))


class TestPriorVals(unittest.TestCase):
    def verify_prior_vals(self, ts, prior_distr):
        span_data = SpansBySamples(ts)
        priors = ConditionalCoalescentTimes(None, prior_distr=prior_distr)
        priors.add(ts.num_samples, approximate=False)
        grid = np.linspace(0, 3, 3)
        mixture_prior = priors.get_mixture_prior_params(span_data)
        nodes_to_date = span_data.nodes_to_date
        prior_vals = fill_prior(mixture_prior, grid, ts, nodes_to_date,
                                prior_distr=prior_distr)
        return prior_vals

    def test_one_tree_n2(self):
        ts = utility_functions.single_tree_ts_n2()
        prior_vals = self.verify_prior_vals(ts, 'gamma')
        self.assertTrue(np.allclose(prior_vals[2], [0, 1, 0.22313016]))

    def test_one_tree_n3(self):
        ts = utility_functions.single_tree_ts_n3()
        prior_vals = self.verify_prior_vals(ts, 'gamma')
        self.assertTrue(np.allclose(prior_vals[3], [0, 1, 0.011109]))
        self.assertTrue(np.allclose(prior_vals[4], [0, 1, 0.3973851]))

    def test_one_tree_n4(self):
        ts = utility_functions.single_tree_ts_n4()
        prior_vals = self.verify_prior_vals(ts, 'gamma')
        self.assertTrue(np.allclose(prior_vals[4], [0, 1, 0.00467134]))
        self.assertTrue(np.allclose(prior_vals[5], [0, 1, 0.02167806]))
        self.assertTrue(np.allclose(prior_vals[6], [0, 1, 0.52637529]))

    def test_polytomy_tree(self):
        ts = utility_functions.polytomy_tree_ts()
        prior_vals = self.verify_prior_vals(ts, 'gamma')
        self.assertTrue(np.allclose(prior_vals[3], [0, 1, 0.3973851]))

    def test_two_tree_ts(self):
        ts = utility_functions.two_tree_ts()
        prior_vals = self.verify_prior_vals(ts, 'gamma')
        self.assertTrue(np.allclose(prior_vals[3], [0, 1, 0.011109]))
        self.assertTrue(np.allclose(prior_vals[4], [0, 1, 0.080002]))
        self.assertTrue(np.allclose(prior_vals[5], [0, 1, 0.3973851]))

    def test_tree_with_unary_nodes(self):
        ts = utility_functions.single_tree_ts_with_unary()
        prior_vals = self.verify_prior_vals(ts, 'gamma')
        self.assertTrue(np.allclose(prior_vals[7], [0, 1, 0.397385]))
        self.assertTrue(np.allclose(prior_vals[6], [0, 1, 0.113122]))
        self.assertTrue(np.allclose(prior_vals[5], [0, 1, 0.164433]))
        self.assertTrue(np.allclose(prior_vals[4], [0, 1, 0.093389]))
        self.assertTrue(np.allclose(prior_vals[3], [0, 1, 0.011109]))


class TestLikelihoodClass(unittest.TestCase):
    def poisson(self, l, x, normalize=True):
        ll = np.exp(-l) * l ** x / scipy.special.factorial(x)
        if normalize:
            return ll / np.max(ll)
        else:
            return ll

    def log_poisson(self, l, x, normalize=True):
        with np.errstate(divide='ignore'):
            ll = np.log(np.exp(-l) * l ** x / scipy.special.factorial(x))
        if normalize:
            return ll - np.max(ll)
        else:
            return ll

    def test_get_mut_edges(self):
        ts = utility_functions.two_tree_mutation_ts()
        mutations_per_edge = Likelihoods.get_mut_edges(ts)
        for e in ts.edges():
            if e.child == 3 and e.parent == 4:
                self.assertEqual(mutations_per_edge[e.id], 2)
            elif e.child == 0 and e.parent == 5:
                self.assertEqual(mutations_per_edge[e.id], 1)
            else:
                self.assertEqual(mutations_per_edge[e.id], 0)

    def test_create_class(self):
        ts = utility_functions.two_tree_mutation_ts()
        grid = np.array([0, 1, 2])
        lik = Likelihoods(ts, grid)
        loglik = LogLikelihoods(ts, grid)
        self.assertRaises(AssertionError, lik.get_mut_lik_fixed_node, ts.edge(0))
        self.assertRaises(AssertionError, lik.get_mut_lik_lower_tri, ts.edge(0))
        self.assertRaises(AssertionError, lik.get_mut_lik_upper_tri, ts.edge(0))
        self.assertRaises(AssertionError, loglik.get_mut_lik_fixed_node, ts.edge(0))
        self.assertRaises(AssertionError, loglik.get_mut_lik_lower_tri, ts.edge(0))
        self.assertRaises(AssertionError, loglik.get_mut_lik_upper_tri, ts.edge(0))

    def test_no_theta_class(self):
        ts = utility_functions.two_tree_mutation_ts()
        grid = np.array([0, 1, 2])
        lik = Likelihoods(ts, grid, theta=None)
        self.assertRaises(RuntimeError, lik.precalculate_mutation_likelihoods)

    def test_precalc_lik_lower(self):
        ts = utility_functions.single_tree_ts_n3()
        grid = np.array([0, 1, 2])
        eps = 0
        theta = 1
        lik = Likelihoods(ts, grid, theta, eps)
        for method in (0, 1, 2):
            # TODO: Remove this loop and hard-code one of the methods after perf testing
            lik.precalculate_mutation_likelihoods(unique_method=method)
            self.assertEquals(ts.num_trees, 1)
            span = ts.first().span
            dt = grid
            num_muts = 0
            n_internal_edges = 0
            expected_lik_dt = self.poisson(dt * (theta / 2 * span), num_muts)
            for edge in ts.edges():
                if ts.node(edge.child).is_sample():
                    self.assertRaises(AssertionError, lik.get_mut_lik_lower_tri, edge)
                    self.assertRaises(AssertionError, lik.get_mut_lik_upper_tri, edge)
                    fixed_edge_lik = lik.get_mut_lik_fixed_node(edge)
                    self.assertTrue(np.allclose(fixed_edge_lik, expected_lik_dt))
                else:
                    n_internal_edges += 1  # only one internal edge in this tree
                    self.assertLessEqual(n_internal_edges, 1)
                    self.assertRaises(AssertionError, lik.get_mut_lik_fixed_node, edge)
                    lower_tri = lik.get_mut_lik_lower_tri(edge)

                    self.assertAlmostEqual(lower_tri[0], expected_lik_dt[0])

                    self.assertAlmostEqual(lower_tri[1], expected_lik_dt[1])
                    self.assertAlmostEqual(lower_tri[2], expected_lik_dt[0])

                    self.assertAlmostEqual(lower_tri[3], expected_lik_dt[2])
                    self.assertAlmostEqual(lower_tri[4], expected_lik_dt[1])
                    self.assertAlmostEqual(lower_tri[5], expected_lik_dt[0])

    def test_precalc_lik_upper_multithread(self):
        ts = utility_functions.two_tree_mutation_ts()
        grid = np.array([0, 1, 2])
        eps = 0
        theta = 1
        for L, pois in [(Likelihoods, self.poisson), (LogLikelihoods, self.log_poisson)]:
            for normalize in (True, False):
                lik = L(ts, grid, theta, eps, normalize=normalize)
                dt = grid
                for num_threads in (None, 1, 2):
                    n_internal_edges = 0
                    lik.precalculate_mutation_likelihoods(num_threads=num_threads)
                    for edge in ts.edges():
                        if not ts.node(edge.child).is_sample():
                            n_internal_edges += 1  # only two internal edges in this tree
                            self.assertLessEqual(n_internal_edges, 2)
                            if edge.parent == 4 and edge.child == 3:
                                num_muts = 2
                            elif edge.parent == 5 and edge.child == 4:
                                num_muts = 0
                            else:
                                self.fail("Unexpected edge")
                            span = edge.right - edge.left
                            expected_lik_dt = pois(
                                dt * (theta / 2 * span), num_muts, normalize=normalize)
                            upper_tri = lik.get_mut_lik_upper_tri(edge)

                            self.assertAlmostEqual(upper_tri[0], expected_lik_dt[0])
                            self.assertAlmostEqual(upper_tri[1], expected_lik_dt[1])
                            self.assertAlmostEqual(upper_tri[2], expected_lik_dt[2])

                            self.assertAlmostEqual(upper_tri[3], expected_lik_dt[0])
                            self.assertAlmostEqual(upper_tri[4], expected_lik_dt[1])

                            self.assertAlmostEqual(upper_tri[5], expected_lik_dt[0])

    def test_tri_functions(self):
        ts = utility_functions.two_tree_mutation_ts()
        grid = np.array([0, 1, 2])
        eps = 0
        theta = 1
        lik = Likelihoods(ts, grid, theta, eps)
        lik.precalculate_mutation_likelihoods()
        for e in ts.edges():
            if e.child == 3 and e.parent == 4:
                exp_branch_muts = 2
                exp_span = 0.2
                self.assertEqual(e.right - e.left, exp_span)
                self.assertEqual(lik.mut_edges[e.id], exp_branch_muts)
                pois_lambda = grid * theta / 2 * exp_span
                cumul_pois = np.cumsum(self.poisson(pois_lambda, exp_branch_muts))
                lower_tri = lik.get_mut_lik_lower_tri(e)
                self.assertTrue(
                    np.allclose(lik.rowsum_lower_tri(lower_tri), cumul_pois))
                upper_tri = lik.get_mut_lik_upper_tri(e)
                self.assertTrue(
                    np.allclose(
                        lik.rowsum_upper_tri(upper_tri)[::-1],
                        cumul_pois))

    def test_no_theta_class_loglikelihood(self):
        ts = utility_functions.two_tree_mutation_ts()
        grid = np.array([0, 1, 2])
        lik = LogLikelihoods(ts, grid, theta=None)
        self.assertRaises(RuntimeError, lik.precalculate_mutation_likelihoods)

    def test_logsumexp(self):
        lls = np.array([0.1, 0.2, 0.5])
        ll_sum = np.sum(lls)
        log_lls = np.log(lls)
        self.assertEqual(LogLikelihoods.logsumexp(log_lls), np.log(ll_sum))

    def test_log_tri_functions(self):
        ts = utility_functions.two_tree_mutation_ts()
        grid = np.array([0, 1, 2])
        eps = 0
        theta = 1
        lik = Likelihoods(ts, grid, theta, eps)
        loglik = LogLikelihoods(ts, grid, theta=theta, eps=eps)
        lik.precalculate_mutation_likelihoods()
        loglik.precalculate_mutation_likelihoods()
        for e in ts.edges():
            if e.child == 3 and e.parent == 4:
                exp_branch_muts = 2
                exp_span = 0.2
                self.assertEqual(e.right - e.left, exp_span)
                self.assertEqual(lik.mut_edges[e.id], exp_branch_muts)
                self.assertEqual(loglik.mut_edges[e.id], exp_branch_muts)
                pois_lambda = grid * theta / 2 * exp_span
                cumul_pois = np.cumsum(self.poisson(pois_lambda, exp_branch_muts))
                lower_tri = lik.get_mut_lik_lower_tri(e)
                lower_tri_log = loglik.get_mut_lik_lower_tri(e)
                self.assertTrue(
                    np.allclose(lik.rowsum_lower_tri(lower_tri), cumul_pois))
                with np.errstate(divide='ignore'):
                    self.assertTrue(
                        np.allclose(loglik.rowsum_lower_tri(lower_tri_log),
                                    np.log(cumul_pois)))
                upper_tri = lik.get_mut_lik_upper_tri(e)
                upper_tri_log = loglik.get_mut_lik_upper_tri(e)
                self.assertTrue(
                    np.allclose(
                        lik.rowsum_upper_tri(upper_tri)[::-1],
                        cumul_pois))
                with np.errstate(divide='ignore'):
                    self.assertTrue(
                        np.allclose(
                            loglik.rowsum_upper_tri(upper_tri_log)[::-1],
                            np.log(cumul_pois)))

    def test_logsumexp_streaming(self):
        lls = np.array([0.1, 0.2, 0.5])
        ll_sum = np.sum(lls)
        log_lls = np.log(lls)
        self.assertTrue(np.allclose(LogLikelihoodsStreaming.logsumexp(log_lls),
                                    np.log(ll_sum)))


class TestNodeGridValuesClass(unittest.TestCase):
    # TODO - needs a few more tests in here
    def test_init(self):
        num_nodes = 5
        ids = np.array([3, 4])
        timepoints = np.array(range(10))
        store = NodeGridValues(num_nodes, ids, timepoints, fill_value=6)
        self.assertEquals(store.grid_data.shape, (len(ids), len(timepoints)))
        self.assertEquals(len(store.fixed_data), (num_nodes-len(ids)))
        self.assertTrue(np.all(store.grid_data == 6))
        self.assertTrue(np.all(store.fixed_data == 6))

        ids = np.array([3, 4], dtype=np.int32)
        store = NodeGridValues(num_nodes, ids, timepoints, fill_value=5)
        self.assertEquals(store.grid_data.shape, (len(ids), len(timepoints)))
        self.assertEquals(len(store.fixed_data), num_nodes-len(ids))
        self.assertTrue(np.all(store.fixed_data == 5))

    def test_set_and_get(self):
        num_nodes = 5
        grid_size = 2
        fill = {}
        for ids in ([3, 4], []):
            np.random.seed(1)
            store = NodeGridValues(
                num_nodes, np.array(ids, dtype=np.int32), np.array(range(grid_size)))
            for i in range(num_nodes):
                fill[i] = np.random.random(grid_size if i in ids else None)
                store[i] = fill[i]
            for i in range(num_nodes):
                self.assertTrue(np.all(fill[i] == store[i]))
        self.assertRaises(IndexError, store.__getitem__, num_nodes)

    def test_bad_init(self):
        ids = [3, 4]
        self.assertRaises(ValueError, NodeGridValues, 3, np.array(ids),
                          np.array([0, 1.2, 2]))
        self.assertRaises(AttributeError, NodeGridValues, 5, np.array(ids), -1)
        self.assertRaises(ValueError, NodeGridValues, 5, np.array([-1]),
                          np.array([0, 1.2, 2]))

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
        self.assertEquals(clone.grid_data.shape, orig.grid_data.shape)
        self.assertEquals(clone.fixed_data.shape, orig.fixed_data.shape)
        self.assertTrue(np.all(clone.grid_data == 0))
        self.assertTrue(np.all(clone.fixed_data == 0))
        # test with something else
        clone = NodeGridValues.clone_with_new_data(orig, 5)
        self.assertEquals(clone.grid_data.shape, orig.grid_data.shape)
        self.assertEquals(clone.fixed_data.shape, orig.fixed_data.shape)
        self.assertTrue(np.all(clone.grid_data == 5))
        self.assertTrue(np.all(clone.fixed_data == 5))
        # test with different
        scalars = np.arange(num_nodes - len(ids))
        clone = NodeGridValues.clone_with_new_data(orig, 0, scalars)
        self.assertEquals(clone.grid_data.shape, orig.grid_data.shape)
        self.assertEquals(clone.fixed_data.shape, orig.fixed_data.shape)
        self.assertTrue(np.all(clone.grid_data == 0))
        self.assertTrue(np.all(clone.fixed_data == scalars))

        clone = NodeGridValues.clone_with_new_data(
            orig, np.array([[1, 2], [4, 3]]))
        for i in range(num_nodes):
            if i in ids:
                self.assertTrue(np.all(clone[i] == orig[i]))
            else:
                self.assertTrue(np.isnan(clone[i]))
        clone = NodeGridValues.clone_with_new_data(
            orig, np.array([[1, 2], [4, 3]]), 0)
        for i in range(num_nodes):
            if i in ids:
                self.assertTrue(np.all(clone[i] == orig[i]))
            else:
                self.assertEquals(clone[i], 0)

    def test_bad_clone(self):
        num_nodes = 10
        ids = [3, 4]
        orig = NodeGridValues(num_nodes, np.array(ids), np.array([0, 1.2]))
        self.assertRaises(
            ValueError,
            NodeGridValues.clone_with_new_data,
            orig, np.array([[1, 2, 3], [4, 5, 6]]))
        self.assertRaises(
            ValueError,
            NodeGridValues.clone_with_new_data,
            orig, 0, np.array([[1, 2], [4, 5]]))


class TestInsideAlgorithm(unittest.TestCase):
    def run_inside_algorithm(self, ts, prior_distr, normalize=True):
        prior = tsdate.build_prior_grid(ts, timepoints=np.array([0, 1.2, 2]),
                                        approximate_prior=False,
                                        prior_distribution=prior_distr)
        theta = 1
        rho = None
        eps = 1e-6
        lls = Likelihoods(ts, prior.timepoints, theta, eps)
        lls.precalculate_mutation_likelihoods()
        algo = InOutAlgorithms(ts, prior, lls)
        algo.inside_pass(theta, rho, normalize=normalize)
        return algo, prior

    def test_one_tree_n2(self):
        ts = utility_functions.single_tree_ts_n2()
        algo = self.run_inside_algorithm(ts, 'gamma')[0]
        self.assertTrue(np.allclose(algo.inside[2], np.array([0, 1, 0.10664654])))

    def test_one_tree_n3(self):
        ts = utility_functions.single_tree_ts_n3()
        algo = self.run_inside_algorithm(ts, 'gamma')[0]
        self.assertTrue(np.allclose(algo.inside[3], np.array([0, 1, 0.0114771635])))
        self.assertTrue(np.allclose(algo.inside[4], np.array([0, 1, 0.1941815518])))

    def test_one_tree_n4(self):
        ts = utility_functions.single_tree_ts_n4()
        algo = self.run_inside_algorithm(ts, 'gamma')[0]
        self.assertTrue(np.allclose(algo.inside[4], np.array([0, 1, 0.00548801])))
        self.assertTrue(np.allclose(algo.inside[5], np.array([0, 1, 0.0239174])))
        self.assertTrue(np.allclose(algo.inside[6], np.array([0, 1, 0.26222197])))

    def test_polytomy_tree(self):
        ts = utility_functions.polytomy_tree_ts()
        algo = self.run_inside_algorithm(ts, 'gamma')[0]
        self.assertTrue(np.allclose(algo.inside[3], np.array([0, 1, 0.12797265])))

    def test_two_tree_ts(self):
        ts = utility_functions.two_tree_ts()
        algo, prior = self.run_inside_algorithm(ts, 'gamma', normalize=False)
        # Prior[3][1] * Ll_(0->3)(1.2 - 0 + eps) ** 2
        node3_t1 = prior[3][1] * scipy.stats.poisson.pmf(
            0, (1.2 + 1e-6) * 0.5 * 0.2) ** 2
        # Prior[3][2] * sum(Ll_(0->3)(2 - t + eps))
        node3_t2 = prior[3][2] * scipy.stats.poisson.pmf(
            0, (2 + 1e-6) * 0.5 * 0.2) ** 2
        self.assertTrue(np.allclose(algo.inside[3],
                                    np.array([0, node3_t1, node3_t2])))
        """
        Prior[4][1] * (Ll_(2->4)(1.2 - 0 + eps) * (Ll_(1->4)(1.2 - 0 + eps)) *
        (Ll_(3->4)(1.2-1.2+eps) * node3_t1)
        """
        node4_t1 = prior[4][1] * (scipy.stats.poisson.pmf(
            0, (1.2 + 1e-6) * 0.5 * 1) * scipy.stats.poisson.pmf(
            0, (1.2 + 1e-6) * 0.5 * 0.8) *
            ((scipy.stats.poisson.pmf(0, (1e-6) * 0.5 * 0.2) * node3_t1)))
        """
        Prior[4][2] * (Ll_(2->4)(2 - 0 + eps) * Ll_(1->4)(2 - 0 + eps) *
        (sum_(t'<2)(Ll_(3->4)(2-t'+eps) * node3_t))
        """
        node4_t2 = prior[4][2] * (scipy.stats.poisson.pmf(
            0, (2 + 1e-6) * 0.5 * 1) * scipy.stats.poisson.pmf(
            0, (2 + 1e-6) * 0.5 * 0.8) * ((scipy.stats.poisson.pmf(
                0, (0.8 + 1e-6) * 0.5 * 0.2) * node3_t1) +
            (scipy.stats.poisson.pmf(0, (1e-6 + 1e-6) * 0.5 * 0.2) * node3_t2)))
        self.assertTrue(np.allclose(algo.inside[4], np.array([0, node4_t1, node4_t2])))
        """
        Prior[5][1] * (Ll_(4->5)(1.2 - 1.2 + eps) * (node3_t ** 0.8)) *
        (Ll_(0->5)(1.2 - 0 + eps) * 1)
        raising node4_t to 0.8 is geometric scaling
        """
        node5_t1 = prior[5][1] * (scipy.stats.poisson.pmf(
            0, (1e-6) * 0.5 * 0.8) * (node4_t1 ** 0.8)) * (scipy.stats.poisson.pmf(
                0, (1.2 + 1e-6) * 0.5 * 0.8))
        """
        Prior[5][2] * (sum_(t'<1.2)(Ll_(4->5)(1.2 - 0 + eps) * (node3_t ** 0.8)) *
        (Ll_(0->5)(1.2 - 0 + eps) * 1)
        """
        node5_t2 = prior[5][2] * ((scipy.stats.poisson.pmf(
            0, (0.8 + 1e-6) * 0.5 * 0.8) * (node4_t1 ** 0.8)) +
            (scipy.stats.poisson.pmf(0, (1e-6 + 1e-6) * 0.5 * 0.8) *
                (node4_t2 ** 0.8))) * (scipy.stats.poisson.pmf(
                    0, (2 + 1e-6) * 0.5 * 0.8))
        self.assertTrue(np.allclose(algo.inside[5], np.array([0, node5_t1, node5_t2])))

    def test_tree_with_unary_nodes(self):
        ts = utility_functions.single_tree_ts_with_unary()
        algo = self.run_inside_algorithm(ts, 'gamma')[0]
        self.assertTrue(np.allclose(algo.inside[7], np.array([0, 1, 0.25406637])))
        self.assertTrue(np.allclose(algo.inside[6], np.array([0, 1, 0.07506923])))
        self.assertTrue(np.allclose(algo.inside[5], np.array([0, 1, 0.13189998])))
        self.assertTrue(np.allclose(algo.inside[4], np.array([0, 1, 0.07370801])))
        self.assertTrue(np.allclose(algo.inside[3], np.array([0, 1, 0.01147716])))

    def test_two_tree_mutation_ts(self):
        ts = utility_functions.two_tree_mutation_ts()
        algo = self.run_inside_algorithm(ts, 'gamma')[0]
        self.assertTrue(np.allclose(algo.inside[3], np.array([0, 1, 0.02176622])))
        # self.assertTrue(np.allclose(upward[4], np.array([0, 2.90560754e-05, 1])))
        # NB the replacement below has not been hand-calculated
        self.assertTrue(np.allclose(algo.inside[4], np.array([0, 3.63200499e-11, 1])))
        # self.assertTrue(np.allclose(upward[5], np.array([0, 5.65044738e-05, 1])))
        # NB the replacement below has not been hand-calculated
        self.assertTrue(np.allclose(algo.inside[5], np.array([0, 7.06320034e-11, 1])))


class TestOutsideAlgorithm(unittest.TestCase):
    def run_outside_algorithm(self, ts, prior_distr="lognorm"):
        span_data = SpansBySamples(ts)
        priors = ConditionalCoalescentTimes(None, prior_distr)
        priors.add(ts.num_samples, approximate=False)
        grid = np.array([0, 1.2, 2])
        mixture_prior = priors.get_mixture_prior_params(span_data)
        nodes_to_date = span_data.nodes_to_date
        prior_vals = fill_prior(
            mixture_prior, grid, ts, nodes_to_date, prior_distr)
        theta = 1
        rho = None
        eps = 1e-6
        lls = Likelihoods(ts, grid, theta, eps)
        lls.precalculate_mutation_likelihoods()
        algo = InOutAlgorithms(ts, prior_vals, lls)
        algo.inside_pass(theta, rho)
        algo.outside_pass(theta, rho, normalize=False)
        return algo

    def test_one_tree_n2(self):
        ts = utility_functions.single_tree_ts_n2()
        for prior_distr in ('lognorm', 'gamma'):
            algo = self.run_outside_algorithm(ts, prior_distr)
            # Root, should this be 0,1,1 or 1,1,1
            self.assertTrue(np.array_equal(
                            algo.outside[2], np.array([1, 1, 1])))

    def test_one_tree_n3(self):
        ts = utility_functions.single_tree_ts_n3()
        for prior_distr in ('lognorm', 'gamma'):
            algo = self.run_outside_algorithm(ts, prior_distr)
            # self.assertTrue(np.allclose(
            #                  downward[3], np.array([0, 1, 0.33508884])))
            self.assertTrue(np.allclose(algo.outside[4], np.array([1, 1, 1])))
            # self.assertTrue(np.allclose(
            #      posterior[3], np.array([0, 0.99616886, 0.00383114])))
            # self.assertTrue(np.allclose(
            #                 posterior[4], np.array([0, 0.83739361, 0.16260639])))

    def test_one_tree_n4(self):
        ts = utility_functions.single_tree_ts_n4()
        for prior_distr in ('lognorm', 'gamma'):
            algo = self.run_outside_algorithm(ts, prior_distr)
            # self.assertTrue(np.allclose(
            #                 downward[4], np.array([0, 1, 0.02187283])))
            # self.assertTrue(np.allclose(
            #                 downward[5], np.array([0, 1, 0.41703272])))
            # Root, should this be 0,1,1 or 1,1,1
            self.assertTrue(np.allclose(
                            algo.outside[6], np.array([1, 1, 1])))

    def test_outside_before_inside_fails(self):
        ts = utility_functions.single_tree_ts_n2()
        prior = tsdate.build_prior_grid(ts)
        theta = 1
        rho = None
        lls = Likelihoods(ts, prior.timepoints, theta)
        lls.precalculate_mutation_likelihoods()
        algo = InOutAlgorithms(ts, prior, lls)
        self.assertRaises(RuntimeError, algo.outside_pass, theta, rho)


class TestTotalFunctionalValueTree(unittest.TestCase):
    """
    Tests to ensure that we recover the total functional value of the tree.
    We can also recover this property in the tree sequence in the special case where
    all node times are known (or all bar one).
    """

    def find_posterior(self, ts, prior_distr):
        span_data = SpansBySamples(ts)
        priors = ConditionalCoalescentTimes(None, prior_distr=prior_distr)
        priors.add(ts.num_samples, approximate=False)
        grid = np.array([0, 1.2, 2])
        mixture_prior = priors.get_mixture_prior_params(span_data)
        nodes_to_date = span_data.nodes_to_date
        prior_vals = fill_prior(
            mixture_prior, grid, ts, nodes_to_date, prior_distr=prior_distr)
        theta = 1
        rho = None
        eps = 1e-6
        lls = Likelihoods(ts, grid, theta, eps)
        lls.precalculate_mutation_likelihoods()
        algo = InOutAlgorithms(ts, prior_vals, lls)
        algo.inside_pass(theta, rho)
        posterior = algo.outside_pass(theta, rho, normalize=False)
        self.assertTrue(np.array_equal(np.sum(
            algo.inside.grid_data * algo.outside.grid_data, axis=1),
            np.sum(algo.inside.grid_data * algo.outside.grid_data, axis=1)))
        self.assertTrue(np.allclose(np.sum(
            algo.inside.grid_data * algo.outside.grid_data, axis=1),
            np.sum(algo.inside.grid_data[-1])))
        return posterior, algo

    def test_one_tree_n2(self):
        ts = utility_functions.single_tree_ts_n2()
        for distr in ('gamma', 'lognorm'):
            posterior, algo = self.find_posterior(ts, distr)

    def test_one_tree_n3(self):
        ts = utility_functions.single_tree_ts_n3()
        for distr in ('gamma', 'lognorm'):
            posterior, algo = self.find_posterior(ts, distr)

    def test_one_tree_n4(self):
        ts = utility_functions.single_tree_ts_n4()
        for distr in ('gamma', 'lognorm'):
            posterior, algo = self.find_posterior(ts, distr)

    def test_one_tree_n3_mutation(self):
        ts = utility_functions.single_tree_ts_mutation_n3()
        for distr in ('gamma', 'lognorm'):
            posterior, algo = self.find_posterior(ts, distr)

    def test_polytomy_tree(self):
        ts = utility_functions.polytomy_tree_ts()
        for distr in ('gamma', 'lognorm'):
            posterior, algo = self.find_posterior(ts, distr)

    def test_tree_with_unary_nodes(self):
        ts = utility_functions.single_tree_ts_with_unary()
        for distr in ('gamma', 'lognorm'):
            posterior, algo = self.find_posterior(ts, distr)


class TestGilTree(unittest.TestCase):
    """
    Test results against hardcoded values Gil independently worked out
    """

    def test_gil_tree(self):
        for cache_inside in [False, True]:
            ts = utility_functions.gils_example_tree()
            span_data = SpansBySamples(ts)
            prior_distr = 'lognorm'
            priors = ConditionalCoalescentTimes(None, prior_distr=prior_distr)
            priors.add(ts.num_samples, approximate=False)
            grid = np.array([0, 0.1, 0.2, 0.5, 1, 2, 5])
            mixture_prior = priors.get_mixture_prior_params(span_data)
            nodes_to_date = span_data.nodes_to_date
            prior_vals = fill_prior(
                mixture_prior, grid, ts, nodes_to_date, prior_distr)
            prior_vals.grid_data[0] = [0, 0.5, 0.3, 0.1, 0.05, 0.02, 0.03]
            prior_vals.grid_data[1] = [0, 0.05, 0.1, 0.2, 0.45, 0.1, 0.1]
            theta = 2
            rho = None
            eps = 0.01
            lls = Likelihoods(ts, grid, theta, eps, normalize=False)
            lls.precalculate_mutation_likelihoods()
            algo = InOutAlgorithms(ts, prior_vals, lls)
            algo.inside_pass(theta, rho, normalize=False, cache_inside=cache_inside)
            algo.outside_pass(theta, rho, normalize=False)
            self.assertTrue(
                np.allclose(np.sum(algo.inside.grid_data * algo.outside.grid_data,
                                   axis=1), [7.44449E-05, 7.44449E-05]))
            self.assertTrue(
                np.allclose(np.sum(algo.inside.grid_data * algo.outside.grid_data,
                                   axis=1), np.sum(algo.inside.grid_data[-1])))


class TestOutsideEdgesOrdering(unittest.TestCase):
    """
    Test that edges_by_child_desc() and edges_by_child_then_parent_desc() order edges
    correctly.
    """

    def edges_ordering(self, ts, fn):
        fixed_node_set = set(ts.samples())
        prior = tsdate.build_prior_grid(ts)
        theta = None
        liklhd = LogLikelihoods(ts, prior.timepoints, theta, 1e-6, fixed_node_set,
                                progress=False)
        dynamic_prog = InOutAlgorithms(ts, prior, liklhd, progress=False)

        if fn == "outside_pass":
            edges_by_child = dynamic_prog.edges_by_child_desc()
            seen_children = list()
            last_child_time = None

            for child, edges in edges_by_child:
                for edge in edges:
                    self.assertTrue(edge.child not in seen_children)
                cur_child_time = ts.tables.nodes.time[child]
                if last_child_time:
                    self.assertTrue(cur_child_time <= last_child_time)
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
                        self.assertTrue(cur_parent_time >= last_parent_time)
                    last_parent_time = cur_parent_time
                self.assertTrue(child not in seen_children)
                cur_child_time = ts.tables.nodes.time[child]
                if last_child_time:
                    self.assertTrue(cur_child_time <= last_child_time)

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
        ts = msprime.simulate(500, Ne=10000, length=5e4, mutation_rate=1e-8,
                              recombination_rate=1e-8)
        sample_data = tsinfer.SampleData.from_tree_sequence(ts, use_times=False)
        inferred_ts = tsinfer.infer(sample_data)
        self.edges_ordering(inferred_ts, "outside_pass")
        self.edges_ordering(inferred_ts, "outside_maximization")


class TestMaximization(unittest.TestCase):
    """
    Test the outside maximization function
    """
    def run_outside_maximization(self, ts, prior_distr="lognorm"):
        prior = tsdate.build_prior_grid(ts, prior_distribution=prior_distr)
        theta = 1
        rho = None
        eps = 1e-6
        lls = Likelihoods(ts, prior.timepoints, theta, eps)
        lls.precalculate_mutation_likelihoods()
        algo = InOutAlgorithms(ts, prior, lls)
        algo.inside_pass(theta, rho)
        return lls, algo, algo.outside_maximization(theta)

    def test_one_tree_n2(self):
        ts = utility_functions.single_tree_ts_n2()
        for prior_distr in ('lognorm', 'gamma'):
            lls, algo, maximized_ages = self.run_outside_maximization(ts, prior_distr)
            self.assertTrue(np.array_equal(
                            maximized_ages,
                            np.array([0, 0, lls.timepoints[np.argmax(algo.inside[2])]])))

    def test_one_tree_n3(self):
        ts = utility_functions.single_tree_ts_n3()
        for prior_distr in ('lognorm', 'gamma'):
            lls, algo, maximized_ages = self.run_outside_maximization(ts, prior_distr)
            node_4 = lls.timepoints[np.argmax(algo.inside[4])]
            ll_mut = scipy.stats.poisson.pmf(
                0, (node_4 - lls.timepoints[:np.argmax(algo.inside[4]) + 1] + 1e-6) *
                1 / 2 * 1)
            result = ll_mut / np.max(ll_mut)
            inside_val = algo.inside[3][:(np.argmax(algo.inside[4]) + 1)]
            node_3 = lls.timepoints[np.argmax(
                result[:np.argmax(algo.inside[4]) + 1] * inside_val)]
            self.assertTrue(np.array_equal(
                            maximized_ages,
                            np.array([0, 0, 0, node_3, node_4])))

    def test_two_tree_ts(self):
        ts = utility_functions.two_tree_ts()
        for prior_distr in ('lognorm', 'gamma'):
            lls, algo, maximized_ages = self.run_outside_maximization(ts, prior_distr)
            node_5 = lls.timepoints[np.argmax(algo.inside[5])]
            ll_mut = scipy.stats.poisson.pmf(
                0, (node_5 - lls.timepoints[:np.argmax(algo.inside[5]) + 1] + 1e-6) *
                1 / 2 * 0.8)
            result = ll_mut / np.max(ll_mut)
            inside_val = algo.inside[4][:(np.argmax(algo.inside[5]) + 1)]
            node_4 = lls.timepoints[np.argmax(
                result[:np.argmax(algo.inside[5]) + 1] * inside_val)]
            ll_mut = scipy.stats.poisson.pmf(
                0, (node_4 - lls.timepoints[:np.argmax(algo.inside[4]) + 1] + 1e-6) *
                1 / 2 * 0.2)
            result = ll_mut / np.max(ll_mut)
            inside_val = algo.inside[3][:(np.argmax(algo.inside[4]) + 1)]
            node_3 = lls.timepoints[np.argmax(
                result[:np.argmax(algo.inside[4]) + 1] * inside_val)]
            self.assertTrue(np.array_equal(
                            maximized_ages,
                            np.array([0, 0, 0, node_3, node_4, node_5])))


class TestDate(unittest.TestCase):
    """
    Test inputs to tsdate.date()
    """
    def test_date_input(self):
        ts = utility_functions.single_tree_ts_n2()
        self.assertRaises(ValueError, tsdate.date, ts, 1, method="foobar")

    def test_sample_as_parent_fails(self):
        ts = utility_functions.single_tree_ts_n3_sample_as_parent()
        self.assertRaises(NotImplementedError, tsdate.date, ts, 1)

    def test_recombination_not_implemented(self):
        ts = utility_functions.single_tree_ts_n2()
        self.assertRaises(NotImplementedError, tsdate.date, ts, 1,
                          recombination_rate=1e-8)


class TestBuildPriorGrid(unittest.TestCase):
    """
    Test tsdate.build_prior_grid() works as expected
    """
    def test_bad_timepoints(self):
        ts = msprime.simulate(2)
        for bad in [-1, np.array([1]), np.array([-1, 2, 3]), np.array([1, 1, 1]),
                    "foobar"]:
            self.assertRaises(ValueError, tsdate.build_prior_grid, ts, timepoints=bad)
        for bad in [np.array(["hello", "there"])]:
            self.assertRaises(TypeError, tsdate.build_prior_grid, ts, timepoints=bad)

    def test_bad_prior_distr(self):
        ts = msprime.simulate(2)
        self.assertRaises(ValueError, tsdate.build_prior_grid, ts,
                          prior_distribution="foobar")


class TestConstrainAgesTopo(unittest.TestCase):
    """
    Test constrain_ages_topo works as expected
    """
    def test_constrain_ages_topo(self):
        """
        Set node 3 to be older than node 4 in two_tree_ts
        """
        ts = utility_functions.two_tree_ts()
        post_mn = np.array([0., 0., 0., 2., 1., 3.])
        timepoints = np.array([0, 1, 2])
        eps = 1e-6
        nodes_to_date = np.array([3, 4, 5])
        constrained_ages = constrain_ages_topo(ts, post_mn, timepoints, eps,
                                               nodes_to_date)
        self.assertTrue(np.array_equal(np.array([0., 0., 0., 2., 2.000001, 3.]),
                                       constrained_ages))

    def test_constrain_ages_topo_no_nodes_to_date(self):
        ts = utility_functions.two_tree_ts()
        post_mn = np.array([0., 0., 0., 2., 1., 3.])
        timepoints = np.array([0, 1, 2])
        eps = 1e-6
        nodes_to_date = None
        constrained_ages = constrain_ages_topo(ts, post_mn, timepoints, eps,
                                               nodes_to_date)
        self.assertTrue(np.array_equal(np.array([0., 0., 0., 2., 2.000001, 3.]),
                                       constrained_ages))
