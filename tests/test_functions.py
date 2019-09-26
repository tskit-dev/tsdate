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
Test cases for the python API for tsdate.
"""
import unittest
import os

import pandas as pd
import tskit  # NOQA
import msprime

import tsdate

import utility_functions


class TestBasicFunctions(unittest.TestCase):
    """
    Test for some of the basic functions used in tsdate
    """
    def test_alpha_prob(self):
        prior = tsdate.prior_maker(10)
        self.assertEqual(prior.m_prob(2, 2, 3), 1.)
        self.assertEqual(prior.m_prob(2, 2, 4), 0.25)

    def test_tau_expect(self):
        prior = tsdate.prior_maker(10)
        self.assertEqual(prior.tau_expect(10, 10), 1.8)
        self.assertEqual(prior.tau_expect(10, 100), 0.09)
        self.assertEqual(prior.tau_expect(100, 100), 1.98)
        self.assertEqual(prior.tau_expect(5, 10), 0.4)

    def test_tau_squared_conditional(self):
        prior = tsdate.prior_maker(10)
        self.assertAlmostEqual(prior.tau_squared_conditional(1, 10), 4.3981418)
        self.assertAlmostEqual(prior.tau_squared_conditional(100, 100), -4.87890977e-18)

    def test_tau_var(self):
        prior = tsdate.prior_maker(10)
        self.assertEqual(prior.tau_var(2, 2), 1)
        self.assertAlmostEqual(prior.tau_var(10, 20), 0.0922995960)
        self.assertAlmostEqual(prior.tau_var(50, 50), 1.15946186)

    def test_gamma_approx(self):
        self.assertEqual(tsdate.gamma_approx(2, 1), (4., 2.))
        self.assertEqual(tsdate.gamma_approx(0.5, 0.1), (2.5, 5.0))

    @unittest.skip("Needs implementing")
    def test_create_time_grid(self):
        raise NotImplementedError


class TestNodeTipWeights(unittest.TestCase):
    def verify_weights(self, ts):
        total_samples, weights_by_node = tsdate.find_node_tip_weights(ts)
        # Check all non-sample nodes in a tree are represented
        nonsample_nodes = set()
        for tree in ts.trees():
            for n in tree.nodes():
                if not tree.is_sample(n):
                    nonsample_nodes.add(n)
        self.assertEqual(set(weights_by_node.keys()), nonsample_nodes)
        for focal_node, tip_weights in weights_by_node.items():
            for num_samples, weights in tip_weights.items():
                self.assertTrue(0 <= focal_node < ts.num_nodes)
                self.assertAlmostEqual(sum(weights.values()), 1.0)
                self.assertLessEqual(max(weights.keys()), ts.num_samples)
        return weights_by_node

    def test_one_tree_n2(self):
        ts = utility_functions.single_tree_ts_n2()
        weights = self.verify_weights(ts)
        # with a single tree there should only be one weight
        for w in weights.values():
            self.assertTrue(len(w), 1)
        self.assertTrue(2 in weights[2][ts.num_samples])  # Root

    def test_one_tree_n3(self):
        ts = utility_functions.single_tree_ts_n3()
        weights = self.verify_weights(ts)
        # with a single tree there should only be one weight
        for w in weights.values():
            self.assertTrue(len(w), 1)
        self.assertTrue(3 in weights[4][ts.num_samples])  # Root
        self.assertTrue(2 in weights[3][ts.num_samples])  # 1st internal node

    def test_one_tree_n4(self):
        ts = utility_functions.single_tree_ts_n4()
        n = ts.num_samples
        weights = self.verify_weights(ts)
        # with a single tree there should only be one weight
        for w in weights.values():
            self.assertTrue(len(w), 1)
        self.assertTrue(4 in weights[6][n])  # Root
        self.assertTrue(3 in weights[5][n])  # 1st internal node
        self.assertTrue(2 in weights[4][n])  # 2nd internal node

    def test_two_trees(self):
        ts = utility_functions.two_tree_ts()
        n = ts.num_samples
        weights = self.verify_weights(ts)
        self.assertEqual(weights[5][n][3], 1.0)  # Root on right tree
        self.assertEqual(weights[4][n][3], 0.2)  # Root on left tree ...
        self.assertEqual(weights[4][n][2], 0.8)  # ... but internal node on right tree
        self.assertEqual(weights[3][n][2], 1.0)  # Internal node on left tree

    def test_missing_tree(self):
        tables = utility_functions.two_tree_ts().tables.keep_intervals(
            [(0, 0.2)], simplify=False)
        ts = tables.tree_sequence()
        n = ts.num_samples
        # Here we have no reference in the trees to node 6
        self.assertRaises(ValueError, tsdate.find_node_tip_weights, ts)
        ts = ts.simplify()
        weights = self.verify_weights(ts)
        self.assertTrue(5 not in weights)    # Root on (deleted) right tree is missin
        self.assertEqual(weights[4][n][3], 1.0)  # Root on left tree ...
        self.assertTrue(2 not in weights[4][n])  # ... but internal on (deleted) r tree
        self.assertEqual(weights[3][n][2], 1.0)  # Internal node on left tree

    def test_tree_with_unary_nodes(self):
        ts = utility_functions.single_tree_ts_with_unary()
        n = ts.num_samples
        weights = self.verify_weights(ts)
        self.assertEqual(weights[3][n][2], 1.0)
        self.assertEqual(weights[4][n][2], 0.5)
        self.assertEqual(weights[4][n][3], 0.5)
        self.assertEqual(weights[5][n][1], 0.5)
        self.assertEqual(weights[5][n][3], 0.5)

    def test_polytomy_tree(self):
        ts = utility_functions.polytomy_tree_ts()
        weights = self.verify_weights(ts)
        self.assertEqual(weights[3][ts.num_samples][3], 1.0)

    def test_larger_find_node_tip_weights(self):
        ts = msprime.simulate(10, recombination_rate=5,
                              mutation_rate=5, random_seed=123)
        self.assertGreater(ts.num_trees, 1)
        self.verify_weights(ts)


class TestMakePrior(unittest.TestCase):
    # We only test make_prior() on single trees
    def verify_prior(self, ts):
        # Check prior contains all possible tips
        prior = tsdate.prior_maker(ts.num_samples)
        prior_df = prior.make_prior()
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

    def test_precalculated_prior(self):
        pm = tsdate.prior_maker(10, approximate=False)
        prior1 = pm.make_prior()
        pm.precalc_approximation_n = 10  # Force approx prior with a tiny n, for testing
        pm.precalculate_prior_for_approximation()
        # Should have created the prior file
        self.assertTrue(os.path.isfile(pm.precalc_approximation_fn))
        prior2 = pm.make_prior()
        self.assertTrue(prior1.equals(prior2))
        # Test when using a bigger n that we are using the precalculated version
        pm100 = tsdate.prior_maker(100, approximate=False)
        pm100.precalc_approximation_n = 10  # Use the same tiny approximation
        # Check here that the approximation is working, by force-reading the approx prior
        # and setting approx = True
        pm100.prior_df = pd.read_csv(pm100.precalc_approximation_fn, index_col=0)
        pm100.approximate = True
        prior100 = pm100.make_prior()
        # Uncomment below to check what we are getting
        # print(prior2)
        # print(prior3)
        # assert False
        self.assertEquals(len(prior100.index), 100)
        pm100.clear_precalculated_prior()
        self.assertFalse(
            os.path.isfile(pm100.precalc_approximation_fn),
            "The file `{}` should have been deleted, but has not been. Please delete it")
