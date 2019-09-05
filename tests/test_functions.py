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
import sys
import tempfile
import pathlib
import unittest
from unittest import mock

import tskit
import msprime
import numpy as np

import tsdate


def single_tree_ts():
    """
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

def polytomy_tree_ts():
    """
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
    """
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
    0       0.1     3       0,1
    0       1       4       2
    0       0.1     4       3
    0.1     1       4       1
    0.1     1       5       0,4
    """)
    return tskit.load_text(nodes=nodes, edges=edges, strict=False)

def single_tree_ts_with_unary():
    """
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
        weights_by_node = tsdate.find_node_tip_weights_ts(ts)
        for n in ts.nodes():
            if not n.is_sample():  # Check all non-sample nodes are represented
                self.assertTrue(n.id in weights_by_node)
        for focal_node, weights in weights_by_node.items():
            self.assertTrue(0 <= focal_node < ts.num_nodes)
            self.assertAlmostEqual(1.0, sum(weights.values()))
            self.assertLessEqual(max(weights.keys()), ts.num_samples)
        return weights_by_node

    def test_one_tree(self):
        ts = single_tree_ts()
        weights = self.verify_weights(ts)
        # with a single tree there should only be one weight
        for w in weights.values():
            self.assertTrue(len(w), 1)
        self.assertTrue(3 in weights[4])  # Root
        self.assertTrue(2 in weights[3])  # 1st internal node

    def test_two_trees(self):
        ts = two_tree_ts()
        weights = self.verify_weights(ts)
        self.assertEqual(weights[5][3], 1.0)  # Root at right
        self.assertEqual(weights[4][3], 0.1)  # Root at left
        self.assertEqual(weights[4][2], 0.9)  # But internal node at right
        self.assertEqual(weights[3][2], 1.0)  # Internal node at left

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
        ts = msprime.simulate(10, recombination_rate=5, mutation_rate=5, random_seed=123)
        self.assertGreater(ts.num_trees, 1)
        weights = self.verify_weights(ts)