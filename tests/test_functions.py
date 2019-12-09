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
import os

import numpy as np
import tskit  # NOQA
import msprime

import tsdate

import utility_functions


class TestBasicFunctions(unittest.TestCase):
    """
    Test for some of the basic functions used in tsdate
    """

    def test_alpha_prob(self):
        self.assertEqual(tsdate.ConditionalCoalescentTimes.m_prob(2, 2, 3), 1.)
        self.assertEqual(tsdate.ConditionalCoalescentTimes.m_prob(2, 2, 4), 0.25)

    def test_tau_expect(self):
        self.assertEqual(tsdate.ConditionalCoalescentTimes.tau_expect(10, 10), 1.8)
        self.assertEqual(tsdate.ConditionalCoalescentTimes.tau_expect(10, 100), 0.09)
        self.assertEqual(tsdate.ConditionalCoalescentTimes.tau_expect(100, 100), 1.98)
        self.assertEqual(tsdate.ConditionalCoalescentTimes.tau_expect(5, 10), 0.4)

    def test_tau_squared_conditional(self):
        self.assertAlmostEqual(
            tsdate.ConditionalCoalescentTimes.tau_squared_conditional(1, 10), 4.3981418)
        self.assertAlmostEqual(
            tsdate.ConditionalCoalescentTimes.tau_squared_conditional(100, 100),
            -4.87890977e-18)

    def test_tau_var(self):
        self.assertEqual(
            tsdate.ConditionalCoalescentTimes.tau_var(2, 2), 1)
        self.assertAlmostEqual(
            tsdate.ConditionalCoalescentTimes.tau_var(10, 20), 0.0922995960)
        self.assertAlmostEqual(
            tsdate.ConditionalCoalescentTimes.tau_var(50, 50), 1.15946186)

    def test_gamma_approx(self):
        self.assertEqual(tsdate.gamma_approx(2, 1), (4., 2.))
        self.assertEqual(tsdate.gamma_approx(0.5, 0.1), (2.5, 5.0))

    @unittest.skip("Needs implementing")
    def test_create_time_grid(self):
        raise NotImplementedError


class TestNodeTipWeights(unittest.TestCase):
    def verify_weights(self, ts):
        span_data = tsdate.SpansBySamples(ts)

        # total_samples, weights_by_node, _ = tsdate.find_node_tip_weights(ts)
        # Check all non-sample nodes in a tree are represented
        nonsample_nodes = set()
        for tree in ts.trees():
            for n in tree.nodes():
                if not tree.is_sample(n):
                    nonsample_nodes.add(n)
        self.assertEqual(set(span_data.nodes_to_date), nonsample_nodes)
        for focal_node in span_data.nodes_to_date:
            for num_samples, weights in span_data.get_weights(focal_node).items():
                self.assertTrue(0 <= focal_node < ts.num_nodes)
                self.assertAlmostEqual(sum(weights.weight), 1.0)
                self.assertLessEqual(max(weights.descendant_tips), ts.num_samples)
        return span_data

    def test_one_tree_n2(self):
        ts = utility_functions.single_tree_ts_n2()
        span_data = self.verify_weights(ts)
        # with a single tree there should only be one weight
        for node in span_data.nodes_to_date:
            self.assertTrue(len(span_data.get_weights(node)), 1)
        self.assertTrue(2 in span_data.get_weights(2)[ts.num_samples])  # Root

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
                np.isin(span_data.get_weights(nd)[n].descendant_tips, expd_tips))

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
                np.isin(span_data.get_weights(nd)[n].descendant_tips, expd_tips))

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
        self.assertRaises(ValueError, tsdate.SpansBySamples, ts)
        ts = ts.simplify()
        span_data = self.verify_weights(ts)
        # Root on (deleted) R tree is missing
        self.assertTrue(5 not in span_data.nodes_to_date)
        self.assertEqual(span_data.lookup_weight(4, n, 3), 1.0)  # Root on L tree ...
        # ... but internal on (deleted) R tree
        self.assertFalse(np.isin(span_data.get_weights(4)[n].descendant_tips, 2))
        self.assertEqual(span_data.lookup_weight(3, n, 2), 1.0)  # Internal nd on L tree

    def test_tree_with_unary_nodes(self):
        ts = utility_functions.single_tree_ts_with_unary()
        n = ts.num_samples
        span_data = self.verify_weights(ts)
        self.assertEqual(span_data.lookup_weight(3, n, 2), 1.0)
        self.assertEqual(span_data.lookup_weight(4, n, 2), 0.5)
        self.assertEqual(span_data.lookup_weight(4, n, 3), 0.5)
        self.assertEqual(span_data.lookup_weight(5, n, 1), 0.5)
        self.assertEqual(span_data.lookup_weight(5, n, 3), 0.5)

    def test_polytomy_tree(self):
        ts = utility_functions.polytomy_tree_ts()
        span_data = self.verify_weights(ts)
        self.assertEqual(span_data.lookup_weight(3, ts.num_samples, 3), 1.0)

    def test_larger_find_node_tip_weights(self):
        ts = msprime.simulate(10, recombination_rate=5,
                              mutation_rate=5, random_seed=123)
        self.assertGreater(ts.num_trees, 1)
        self.verify_weights(ts)


class TestMakePrior(unittest.TestCase):
    # We only test make_prior() on single trees
    def verify_prior(self, ts):
        # Check prior contains all possible tips
        priors = tsdate.ConditionalCoalescentTimes(None)  # Don't use approximation
        priors.add(ts.num_samples)
        prior_df = priors[ts.num_samples]
        self.assertEqual(prior_df.shape[0], ts.num_samples)
        return(prior_df)

    def test_one_tree_n2(self):
        ts = utility_functions.single_tree_ts_n2()
        prior = self.verify_prior(ts)
        [self.assertAlmostEqual(x, y)
         for x, y in zip(prior.loc[2].values,
                         [1., 1.])]

    def test_one_tree_n3(self):
        ts = utility_functions.single_tree_ts_n3()
        prior = self.verify_prior(ts)
        [self.assertAlmostEqual(x, y)
         for x, y in zip(prior.loc[2].values,
                         [1., 3.])]
        [self.assertAlmostEqual(x, y)
         for x, y in zip(prior.loc[3].values,
                         [1.6, 1.2])]

    def test_one_tree_n4(self):
        ts = utility_functions.single_tree_ts_n4()
        prior = self.verify_prior(ts)
        [self.assertAlmostEqual(x, y)
         for x, y in zip(prior.loc[2].values,
                         [0.81818182, 3.27272727])]
        [self.assertAlmostEqual(x, y)
         for x, y in zip(prior.loc[3].values,
                         [1.8, 3.6])]
        [self.assertAlmostEqual(x, y)
         for x, y in zip(prior.loc[4].values,
                         [1.97560976, 1.31707317])]

    def test_polytomy_tree(self):
        ts = utility_functions.polytomy_tree_ts()
        prior = self.verify_prior(ts)
        [self.assertAlmostEqual(x, y)
         for x, y in zip(prior.loc[3].values,
                         [1.6, 1.2])]

    def test_two_tree_ts(self):
        ts = utility_functions.two_tree_ts()
        prior = self.verify_prior(ts)
        [self.assertAlmostEqual(x, y)
         for x, y in zip(prior.loc[2].values,
                         [1., 3.])]
        [self.assertAlmostEqual(x, y)
         for x, y in zip(prior.loc[3].values,
                         [1.6, 1.2])]

    def test_single_tree_ts_with_unary(self):
        ts = utility_functions.single_tree_ts_with_unary()
        prior = self.verify_prior(ts)
        [self.assertAlmostEqual(x, y)
         for x, y in zip(prior.loc[2].values,
                         [1., 3.])]
        [self.assertAlmostEqual(x, y)
         for x, y in zip(prior.loc[3].values,
                         [1.6, 1.2])]

    def test_two_tree_mutation_ts(self):
        ts = utility_functions.two_tree_mutation_ts()
        prior = self.verify_prior(ts)
        [self.assertAlmostEqual(x, y)
         for x, y in zip(prior.loc[2].values,
                         [1., 3.])]
        [self.assertAlmostEqual(x, y)
         for x, y in zip(prior.loc[3].values,
                         [1.6, 1.2])]

    def test_precalculated_prior(self):
        # Force approx prior with a tiny n
        priors_approx10 = tsdate.ConditionalCoalescentTimes(10)
        priors_approx10.add(10)
        # Check we have created the prior file
        self.assertTrue(
            os.path.isfile(tsdate.ConditionalCoalescentTimes.precalc_approx_fn(10)))
        priors_approxNone = tsdate.ConditionalCoalescentTimes(None)
        priors_approxNone.add(10)
        self.assertTrue(priors_approx10[10].equals(priors_approxNone[10]))
        # Test when using a bigger n that we're using the precalculated version
        priors_approx10.add(100)
        self.assertEquals(len(priors_approx10[100].index), 100)
        priors_approxNone.add(100, approximate=False)
        self.assertEquals(len(priors_approxNone[100].index), 100)
        self.assertFalse(priors_approx10[100].equals(priors_approxNone[100]))

        priors_approx10.clear_precalculated_prior()
        self.assertFalse(
            os.path.isfile(tsdate.ConditionalCoalescentTimes.precalc_approx_fn(10)),
            "The file `{}` should have been deleted, but has not been.\
             Please delete it")


class TestMixturePrior(unittest.TestCase):
    def get_mixture_prior(self, ts):
        span_data = tsdate.SpansBySamples(ts)
        priors = tsdate.ConditionalCoalescentTimes(None)
        priors.add(ts.num_samples, approximate=False)
        mixture_prior = tsdate.get_mixture_prior(span_data, priors)
        return(mixture_prior)

    def test_one_tree_n2(self):
        ts = utility_functions.single_tree_ts_n2()
        mixture_prior = self.get_mixture_prior(ts)
        [self.assertAlmostEqual(x, y)
         for x, y in zip(mixture_prior.loc[2].values,
                         [1., 1.])]

    def test_one_tree_n3(self):
        ts = utility_functions.single_tree_ts_n3()
        mixture_prior = self.get_mixture_prior(ts)
        [self.assertAlmostEqual(x, y)
         for x, y in zip(mixture_prior.loc[3].values,
                         [1., 3.])]
        [self.assertAlmostEqual(x, y)
         for x, y in zip(mixture_prior.loc[4].values,
                         [1.6, 1.2])]

    def test_one_tree_n4(self):
        ts = utility_functions.single_tree_ts_n4()
        mixture_prior = self.get_mixture_prior(ts)
        [self.assertAlmostEqual(x, y)
         for x, y in zip(mixture_prior.loc[4].values,
                         [0.81818182, 3.27272727])]
        [self.assertAlmostEqual(x, y)
         for x, y in zip(mixture_prior.loc[5].values,
                         [1.8, 3.6])]
        [self.assertAlmostEqual(x, y)
         for x, y in zip(mixture_prior.loc[6].values,
                         [1.97560976, 1.31707317])]

    def test_polytomy_tree(self):
        ts = utility_functions.polytomy_tree_ts()
        mixture_prior = self.get_mixture_prior(ts)
        [self.assertAlmostEqual(x, y)
         for x, y in zip(mixture_prior.loc[3].values,
                         [1.6, 1.2])]

    def test_two_trees(self):
        ts = utility_functions.two_tree_ts()
        mixture_prior = self.get_mixture_prior(ts)
        [self.assertAlmostEqual(x, y)
         for x, y in zip(mixture_prior.loc[3].values,
                         [1., 3.])]
        # Node 4 should be a mixture between 2 and 3 tips
        [self.assertAlmostEqual(x, y, places=4)
         for x, y in zip(mixture_prior.loc[4].values,
                         [0.60377, 1.13207])]
        [self.assertAlmostEqual(x, y)
         for x, y in zip(mixture_prior.loc[5].values,
                         [1.6, 1.2])]

    def test_single_tree_ts_with_unary(self):
        ts = utility_functions.single_tree_ts_with_unary()
        mixture_prior = self.get_mixture_prior(ts)
        [self.assertAlmostEqual(x, y)
         for x, y in zip(mixture_prior.loc[3].values,
                         [1., 3.])]
        # Node 4 should be a mixture between 2 and 3 tips
        [self.assertAlmostEqual(x, y, places=4)
         for x, y in zip(mixture_prior.loc[4].values,
                         [0.80645, 0.96774])]
        # Node 5 should be a mixture between 1 and 3 tips
        [self.assertAlmostEqual(x, y, places=4)
         for x, y in zip(mixture_prior.loc[5].values,
                         [0.44444, 0.66666])]
        [self.assertAlmostEqual(x, y)
            for x, y in zip(mixture_prior.loc[6].values,
                            [1.6, 1.2])]

    def test_two_tree_mutation_ts(self):
        ts = utility_functions.two_tree_mutation_ts()
        mixture_prior = self.get_mixture_prior(ts)
        [self.assertAlmostEqual(x, y)
         for x, y in zip(mixture_prior.loc[3].values,
                         [1., 3.])]
        # Node 4 should be a mixture between 2 and 3 tips
        [self.assertAlmostEqual(x, y, places=4)
         for x, y in zip(mixture_prior.loc[4].values,
                         [0.60377, 1.13207])]
        [self.assertAlmostEqual(x, y)
         for x, y in zip(mixture_prior.loc[5].values,
                         [1.6, 1.2])]


class TestPriorVals(unittest.TestCase):
    def verify_prior_vals(self, ts):
        span_data = tsdate.SpansBySamples(ts)
        priors = tsdate.ConditionalCoalescentTimes(None)
        priors.add(ts.num_samples, approximate=False)
        grid = np.linspace(0, 3, 3)
        mixture_prior = tsdate.get_mixture_prior(span_data, priors)
        nodes_to_date = span_data.nodes_to_date
        prior_vals = tsdate.get_prior_values(mixture_prior, grid, ts, nodes_to_date)
        self.assertTrue(np.array_equal(prior_vals[0:ts.num_samples],
                        np.tile(np.array([1, 0, 0]), (ts.num_samples, 1))))
        return prior_vals

    def test_one_tree_n2(self):
        ts = utility_functions.single_tree_ts_n2()
        prior_vals = self.verify_prior_vals(ts)
        self.assertTrue(np.allclose(prior_vals[2],
                        np.array([0, 1, 0.22313016])))

    def test_one_tree_n3(self):
        ts = utility_functions.single_tree_ts_n3()
        prior_vals = self.verify_prior_vals(ts)
        self.assertTrue(np.allclose(prior_vals[3],
                        np.array([0, 1, 0.011109])))
        self.assertTrue(np.allclose(prior_vals[4],
                        np.array([0, 1, 0.3973851])))

    def test_one_tree_n4(self):
        ts = utility_functions.single_tree_ts_n4()
        prior_vals = self.verify_prior_vals(ts)
        self.assertTrue(np.allclose(prior_vals[4],
                        np.array([0, 1, 0.00467134])))
        self.assertTrue(np.allclose(prior_vals[5],
                        np.array([0, 1, 0.02167806])))
        self.assertTrue(np.allclose(prior_vals[6],
                        np.array([0, 1, 0.52637529])))

    def test_polytomy_tree(self):
        ts = utility_functions.polytomy_tree_ts()
        prior_vals = self.verify_prior_vals(ts)
        self.assertTrue(np.allclose(prior_vals[3],
                        np.array([0, 1, 0.3973851])))

    def test_two_tree_ts(self):
        ts = utility_functions.two_tree_ts()
        prior_vals = self.verify_prior_vals(ts)
        self.assertTrue(np.allclose(prior_vals[3],
                        np.array([0, 1, 0.011109])))
        self.assertTrue(np.allclose(prior_vals[4],
                        np.array([0, 1, 0.080002])))
        self.assertTrue(np.allclose(prior_vals[5],
                        np.array([0, 1, 0.3973851])))

    def test_tree_with_unary_nodes(self):
        ts = utility_functions.single_tree_ts_with_unary()
        prior_vals = self.verify_prior_vals(ts)
        self.assertTrue(np.allclose(prior_vals[3],
                        np.array([0, 1, 0.011109])))
        self.assertTrue(np.allclose(prior_vals[4],
                        np.array([0, 1, 0.16443276])))
        self.assertTrue(np.allclose(prior_vals[5],
                        np.array([0, 1, 0.11312131])))
        self.assertTrue(np.allclose(prior_vals[6],
                        np.array([0, 1, 0.3973851])))


class TestForwardAlgorithm(unittest.TestCase):
    def verify_forward_algorithm(self, ts):
        span_data = tsdate.SpansBySamples(ts)
        priors = tsdate.ConditionalCoalescentTimes(None)
        priors.add(ts.num_samples, approximate=False)
        grid = np.array([0, 1.2, 2])
        mixture_prior = tsdate.get_mixture_prior(span_data, priors)
        nodes_to_date = span_data.nodes_to_date
        prior_vals = tsdate.get_prior_values(mixture_prior, grid, ts, nodes_to_date)
        theta = 1
        rho = None
        eps = 1e-6
        lls = tsdate.Likelihoods(ts, grid, eps)
        lls.precalculate_mutation_likelihoods(theta)
        forward, g_i, logged_forwards, logged_g_i = \
            tsdate.forward_algorithm(
                ts, prior_vals, theta, rho, lls, False)
        self.assertTrue(np.array_equal(forward[0:ts.num_samples],
                        np.tile(np.array([1, 0, 0]), (ts.num_samples, 1))))
        self.assertTrue(np.allclose(logged_forwards[0:ts.num_samples],
                        np.tile(np.array([1e-10, -23.02585, -23.02585]),
                                (ts.num_samples, 1))))
        self.assertTrue(np.allclose(logged_g_i[0:ts.num_samples],
                        np.tile(np.array([1e-10, -23.02585, -23.02585]),
                                (ts.num_samples, 1))))
        return forward, logged_g_i

    def test_one_tree_n2(self):
        ts = utility_functions.single_tree_ts_n2()
        forward, logged_g_i = self.verify_forward_algorithm(ts)
        self.assertTrue(np.allclose(forward[2], np.array([0, 1, 0.10664654])))

    def test_one_tree_n3(self):
        ts = utility_functions.single_tree_ts_n3()
        forward, logged_g_i = self.verify_forward_algorithm(ts)
        self.assertTrue(np.allclose(forward[3],
                        np.array([0, 1, 0.0114771635])))
        self.assertTrue(np.allclose(forward[4],
                        np.array([0, 1, 0.1941815518])))

    def test_one_tree_n4(self):
        ts = utility_functions.single_tree_ts_n4()
        forward, logged_g_i = self.verify_forward_algorithm(ts)
        self.assertTrue(np.allclose(forward[4], np.array([0, 1, 0.00548801])))
        self.assertTrue(np.allclose(forward[5], np.array([0, 1, 0.0239174])))
        self.assertTrue(np.allclose(forward[6], np.array([0, 1, 0.26222197])))

    def test_polytomy_tree(self):
        ts = utility_functions.polytomy_tree_ts()
        forward, logged_g_i = self.verify_forward_algorithm(ts)
        self.assertTrue(np.allclose(forward[3], np.array([0, 1, 0.12797265])))

    def test_two_tree_ts(self):
        ts = utility_functions.two_tree_ts()
        forward, logged_g_i = self.verify_forward_algorithm(ts)
        self.assertTrue(np.allclose(forward[3], np.array([0, 1, 0.02176622])))
        self.assertTrue(np.allclose(forward[4], np.array([0, 1, 0.04403458])))
        self.assertTrue(np.allclose(forward[5], np.array([0, 1, 0.23762418])))

    def test_tree_with_unary_nodes(self):
        ts = utility_functions.single_tree_ts_with_unary()
        forward, logged_g_i = self.verify_forward_algorithm(ts)
        self.assertTrue(np.allclose(forward[3], np.array([0, 1, 0.01147716])))
        self.assertTrue(np.allclose(forward[4], np.array([0, 1, 0.12086781])))
        self.assertTrue(np.allclose(forward[5], np.array([0, 1, 0.07506923])))
        self.assertTrue(np.allclose(forward[6], np.array([0, 1, 0.25057244])))

    def test_two_tree_mutation_ts(self):
        ts = utility_functions.two_tree_mutation_ts()
        forward, logged_g_i = self.verify_forward_algorithm(ts)
        self.assertTrue(np.allclose(forward[3], np.array([0, 1, 0.02176622])))
        self.assertTrue(np.allclose(forward[4],
                        np.array([0, 2.90560754e-05, 1])))
        self.assertTrue(np.allclose(forward[5],
                        np.array([0, 5.65044738e-05, 1])))


class TestBackwardAlgorithm(unittest.TestCase):
    def verify_backward_algorithm(self, ts):
        fixed_nodes_set = set(ts.samples())
        span_data = tsdate.SpansBySamples(ts)
        spans = span_data.node_total_span
        priors = tsdate.ConditionalCoalescentTimes(None)
        priors.add(ts.num_samples, approximate=False)
        grid = np.array([0, 1.2, 2])
        mixture_prior = tsdate.get_mixture_prior(span_data, priors)
        nodes_to_date = span_data.nodes_to_date
        prior_vals = tsdate.get_prior_values(mixture_prior, grid, ts, nodes_to_date)
        theta = 1
        rho = None
        eps = 1e-6
        lls = tsdate.Likelihoods(ts, grid, eps)
        lls.precalculate_mutation_likelihoods(theta)
        forward, g_i, logged_forwards, logged_g_i = \
            tsdate.forward_algorithm(
                ts, prior_vals, theta, rho, lls, False)
        posterior, backward = \
            tsdate.backward_algorithm(
                ts, logged_forwards, logged_g_i, theta, rho,
                lls, spans, fixed_nodes_set)
        self.assertTrue(np.array_equal(backward[0:ts.num_samples],
                        np.tile(np.array([1, 0, 0]), (ts.num_samples, 1))))
        self.assertTrue(np.array_equal(posterior[0:ts.num_samples],
                        np.tile(np.array([1, 0, 0]), (ts.num_samples, 1))))
        return posterior, backward

    def test_one_tree_n2(self):
        ts = utility_functions.single_tree_ts_n2()
        posterior, backward = self.verify_backward_algorithm(ts)
        self.assertTrue(np.array_equal(
                        backward[2], np.array([0, 1, 1])))

    def test_one_tree_n3(self):
        ts = utility_functions.single_tree_ts_n3()
        posterior, backward = self.verify_backward_algorithm(ts)
        self.assertTrue(np.allclose(
                         backward[3], np.array([0, 1, 0.33508884])))
        self.assertTrue(np.allclose(backward[4], np.array([0, 1, 1])))
        self.assertTrue(np.allclose(
             posterior[3], np.array([0, 0.99616886, 0.00383114])))
        self.assertTrue(np.allclose(
                        posterior[4], np.array([0, 0.83739361, 0.16260639])))

    def test_one_tree_n4(self):
        ts = utility_functions.single_tree_ts_n4()
        posterior, backward = self.verify_backward_algorithm(ts)
        self.assertTrue(np.allclose(
                        backward[4], np.array([0, 1, 0.02187283])))
        self.assertTrue(np.allclose(
                        backward[5], np.array([0, 1, 0.41703272])))
        self.assertTrue(np.allclose(
                        backward[6], np.array([0, 1, 1])))
