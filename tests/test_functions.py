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
import io
import unittest

import tskit
import msprime

import tsdate


def single_tree_ts_n2():
    r"""
    Simple case where we have n = 2 and one tree.
         2
        / \
       0   1
    """
    nodes = io.StringIO("""\
    id      is_sample   time
    0       1           0
    1       1           0
    2       0           1
    """)
    edges = io.StringIO("""\
    left    right   parent  child
    0       1       2       0,1
    """)
    return(tskit.load_text(nodes=nodes, edges=edges, strict=False))


def single_tree_ts_n3():
    r"""
    Simple case where we have n = 3 and one tree.
            4
           / \
          3   \
         / \   \
        0   1   2
    """
    nodes = io.StringIO("""\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       0           1
    4       0           2
    """)
    edges = io.StringIO("""\
    left    right   parent  child
    0       1       3       0,1
    0       1       4       2,3
    """)
    return tskit.load_text(nodes=nodes, edges=edges, strict=False)


def single_tree_ts_n4():
    r"""
    Simple case where we have n = 4 and one tree.
              6
             / \
            5   \
           / \   \
          4   \   \
         / \   \   \
        0   1   2   3
    """
    nodes = io.StringIO("""\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       1           0
    4       0           1
    5       0           2
    6       0           3
    """)
    edges = io.StringIO("""\
    left    right   parent  child
    0       1       4       0,1
    0       1       5       2,4
    0       1       6       3,5
    """)
    return tskit.load_text(nodes=nodes, edges=edges, strict=False)


def polytomy_tree_ts():
    r"""
    Simple case where we have n = 3 and a polytomy.
          3
         /|\
        0 1 2
    """
    nodes = io.StringIO("""\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       0           1
    """)
    edges = io.StringIO("""\
    left    right   parent  child
    0       1       3       0,1,2
    """)
    return tskit.load_text(nodes=nodes, edges=edges, strict=False)


def two_tree_ts():
    r"""
    Simple case where we have n = 3 and 2 trees.
                   .    5
                   .   / \
            4      .  |   4
           / \     .  |   |\
          3   \    .  |   | \
         / \   \   .  |   |  \
        0   1   2  .  0   1   2
    """
    nodes = io.StringIO("""\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       0           1
    4       0           2
    5       0           3
    """)
    edges = io.StringIO("""\
    left    right   parent  child
    0       0.2     3       0,1
    0       1       4       2
    0       0.2     4       3
    0.2     1       4       1
    0.2     1       5       0,4
    """)
    return tskit.load_text(nodes=nodes, edges=edges, strict=False)


def single_tree_ts_with_unary():
    r"""
    Simple case where we have n = 3 and some unary nodes.
            6
           / 5
          4   \
          3    \
         / \    \
        0   1    2
    """
    nodes = io.StringIO("""\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       0           1
    4       0           2
    5       0           3
    6       0           4
    """)
    edges = io.StringIO("""\
    left    right   parent  child
    0       1       3       0,1
    0       1       5       2
    0       1       4       3
    0       1       6       4,5
    """)
    return tskit.load_text(nodes=nodes, edges=edges, strict=False)


class TestBasicFunctions(unittest.TestCase):
    """
    Test for some of the basic functions used in tsdate
    """
    @unittest.skip("Needs implementing")
    def test_alpha_prob(self):
        raise NotImplementedError

    @unittest.skip("Needs implementing")
    def test_tau_expect(self):
        raise NotImplementedError

    @unittest.skip("Needs implementing")
    def test_expect_tau_cond_alpha(self):
        raise NotImplementedError

    @unittest.skip("Needs implementing")
    def test_tau_squared_conditional(self):
        raise NotImplementedError

    @unittest.skip("Needs implementing")
    def test_tau_var(self):
        raise NotImplementedError

    @unittest.skip("Needs implementing")
    def test_gamma_approx(self):
        raise NotImplementedError

    @unittest.skip("Needs implementing")
    def test_make_prior(self):
        raise NotImplementedError

    @unittest.skip("Needs implementing")
    def test_create_time_grid(self):
        raise NotImplementedError


class TestNodeTipWeights(unittest.TestCase):
    def verify_weights(self, ts):
        weights_by_node = tsdate.find_node_tip_weights(ts)
        # Check all internal nodes in a tree are represented
        internal_nodes = set()
        for tree in ts.trees():
            for n in tree.nodes():
                if tree.is_internal(n):
                    internal_nodes.add(n)
        self.assertEqual(set(weights_by_node.keys()), internal_nodes)
        for focal_node, weights in weights_by_node.items():
            self.assertTrue(0 <= focal_node < ts.num_nodes)
            self.assertAlmostEqual(sum(weights.values()), 1.0)
            self.assertLessEqual(max(weights.keys()), ts.num_samples)
        return weights_by_node

    def test_one_tree_n2(self):
        ts = single_tree_ts_n2()
        weights = self.verify_weights(ts)
        # with a single tree there should only be one weight
        for w in weights.values():
            self.assertTrue(len(w), 1)
        self.assertTrue(2 in weights[2])  # Root

    def test_one_tree_n3(self):
        ts = single_tree_ts_n3()
        weights = self.verify_weights(ts)
        # with a single tree there should only be one weight
        for w in weights.values():
            self.assertTrue(len(w), 1)
        self.assertTrue(3 in weights[4])  # Root
        self.assertTrue(2 in weights[3])  # 1st internal node

    def test_one_tree_n4(self):
        ts = single_tree_ts_n4()
        weights = self.verify_weights(ts)
        # with a single tree there should only be one weight
        for w in weights.values():
            self.assertTrue(len(w), 1)
        self.assertTrue(4 in weights[6])  # Root
        self.assertTrue(3 in weights[5])  # 1st internal node
        self.assertTrue(2 in weights[4])  # 2nd internal node

    def test_two_trees(self):
        ts = two_tree_ts()
        weights = self.verify_weights(ts)
        self.assertEqual(weights[5][3], 1.0)  # Root on right tree
        self.assertEqual(weights[4][3], 0.2)  # Root on left tree ...
        self.assertEqual(weights[4][2], 0.8)  # ... but internal node on right tree
        self.assertEqual(weights[3][2], 1.0)  # Internal node on left tree

    def test_missing_tree(self):
        tables = two_tree_ts().tables.keep_intervals([(0, 0.2)], simplify=False)
        ts = tables.tree_sequence()
        weights = self.verify_weights(ts)
        self.assertTrue(5 not in weights)    # Root on (deleted) right tree
        self.assertEqual(weights[4][3], 1.0)  # Root on left tree ...
        self.assertTrue(2 not in weights[4])  # ... but internal node on (deleted) r tree
        self.assertEqual(weights[3][2], 1.0)  # Internal node on left tree

    def test_tree_with_unary_nodes(self):
        ts = single_tree_ts_with_unary()
        weights = self.verify_weights(ts)
        self.assertEqual(weights[3][2], 1.0)
        self.assertEqual(weights[4][2], 1.0)
        self.assertEqual(weights[5][1], 1.0)

    def test_polytomy_tree(self):
        ts = polytomy_tree_ts()
        weights = self.verify_weights(ts)
        self.assertEqual(weights[3][3], 1.0)

    def test_larger_find_node_tip_weights(self):
        ts = msprime.simulate(10, recombination_rate=5,
                              mutation_rate=5, random_seed=123)
        self.assertGreater(ts.num_trees, 1)
        self.verify_weights(ts)


class TestMakePrior(unittest.TestCase):
    # We only test make_prior() on single trees
    def verify_prior(self, ts):
        # Check prior contains all possible tips
        prior = tsdate.make_prior(n=ts.num_samples)
        self.assertEqual(prior.shape[0], ts.num_samples)
        self.assertTrue(prior["Expected_Age"].iloc[-1] <= 2.0)
        return(prior)

    def test_one_tree_n2(self):
        ts = single_tree_ts_n2()
        prior = self.verify_prior(ts)
        [self.assertAlmostEqual(x, y)
         for x, y in zip(prior.loc[2].values,
                         [1., 1., 1., 1.])]

    def test_one_tree_n3(self):
        ts = single_tree_ts_n3()
        prior = self.verify_prior(ts)
        [self.assertAlmostEqual(x, y)
         for x, y in zip(prior.loc[2].values,
                         [0.33333333, 0.11111111, 1., 3.])]
        [self.assertAlmostEqual(x, y)
         for x, y in zip(prior.loc[3].values,
                         [1.33333333, 1.11111111, 1.6, 1.2])]

    def test_one_tree_n4(self):
        ts = single_tree_ts_n4()
        prior = self.verify_prior(ts)
        [self.assertAlmostEqual(x, y)
         for x, y in zip(prior.loc[2].values,
                         [0.25, 0.07638889, 0.81818182, 3.27272727])]
        [self.assertAlmostEqual(x, y)
         for x, y in zip(prior.loc[3].values,
                         [0.5, 0.13888889, 1.8, 3.6])]
        [self.assertAlmostEqual(x, y)
         for x, y in zip(prior.loc[4].values,
                         [1.5, 1.13888889, 1.97560976, 1.31707317])]

    def test_polytomy_tree(self):
        ts = polytomy_tree_ts()
        prior = self.verify_prior(ts)
        [self.assertAlmostEqual(x, y)
         for x, y in zip(prior.loc[3].values,
                         [1.33333333, 1.11111111, 1.6, 1.2])]

