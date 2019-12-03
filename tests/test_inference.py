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
import unittest

import numpy as np
import tskit
import msprime
import tsinfer

import tsdate
import utility_functions


# class TestPrebuilt(unittest.TestCase):
#     """
#     Tests for tsdate on prebuilt tree sequences
#     """
#     def test_simple_ts_n2(self):
#         ts = utility_functions.single_tree_ts_n2()
#         dated_ts = tsdate.date(ts, Ne=10000)
#         self.assertTrue(np.array_equal(dated_ts.tables.nodes.time[:],
#                         np.array([0., 0., 0.01])))

#     def test_simple_ts_n3(self):
#         ts = utility_functions.single_tree_ts_n3()
#         dated_ts = tsdate.date(ts, Ne=10000)
#         self.assertTrue(np.allclose(dated_ts.tables.nodes.time[:],
#                         np.array([0.00000000e+00, 0.00000000e+00,
#                                   0.00000000e+00, 1.00000000e-02,
#                                   3.44872693e+03])))

#     def test_simple_ts_n4(self):
#         ts = utility_functions.single_tree_ts_n4()
#         dated_ts = tsdate.date(ts, Ne=10000)
#         self.assertTrue(np.allclose(dated_ts.tables.nodes.time[:],
#                         np.array([0.00000000e+00, 0.00000000e+00,
#                                   0.00000000e+00, 0.00000000e+00,
#                                   1.00000000e-02, 2.59235395e+03,
#                                   6.04463820e+03])))

#     def test_polytomy_ts(self):
#         ts = utility_functions.polytomy_tree_ts()
#         dated_ts = tsdate.date(ts, Ne=10000)
#         self.assertTrue(np.array_equal(dated_ts.tables.nodes.time[:],
#                         np.array([0., 0., 0., 0.01])))

#     def test_two_tree_ts(self):
#         ts = utility_functions.two_tree_ts()
#         dated_ts = tsdate.date(ts, Ne=10000)
#         self.assertTrue(np.allclose(dated_ts.tables.nodes.time[:],
#                         np.array([0.00000000e+00, 0.00000000e+00,
#                                   0.00000000e+00, 1.00000000e-02,
#                                   3.44872693e+03, 8.58901551e+03])))

#     def test_single_tree_ts_unary(self):
#         ts = utility_functions.single_tree_ts_with_unary()
#         dated_ts = tsdate.date(ts, Ne=10000)
#         self.assertTrue(np.allclose(dated_ts.tables.nodes.time[:],
#                         np.array([0.00000000e+00, 0.00000000e+00,
#                                   0.00000000e+00, 1.00000000e-02,
#                                   3.44872693e+03, 1.09851474e+04,
#                                   1.09851574e+04])))

#     def test_two_tree_mutation_ts(self):
#         ts = utility_functions.two_tree_mutation_ts()
#         dated_ts = tsdate.date(ts, Ne=10000, mutation_rate=1e-8)
#         self.assertTrue(np.allclose(dated_ts.tables.nodes.time[:],
#                         np.array([0.00000000e+00, 0.00000000e+00,
#                                   0.00000000e+00, 1.00000000e-02,
#                                   3.44863243e+03, 1.63560039e+04])))


class TestSimulated(unittest.TestCase):
    """
    Tests for tsdate on simulated tree sequences.
    """
    def ts_equal_except_times(self, ts1, ts2):
        for (t1_name, t1), (t2_name, t2) in zip(ts1.tables, ts2.tables):
            if isinstance(t1, tskit.ProvenanceTable):
                # TO DO - should check that the provenance has had the "tsdate" method
                # added
                pass
            elif isinstance(t1, tskit.NodeTable):
                for column_name in t1.column_names:
                    if column_name != 'time':
                        col_t1 = getattr(t1, column_name)
                        col_t2 = getattr(t2, column_name)
                        self.assertTrue(np.array_equal(col_t1, col_t2))
            elif isinstance(t1, tskit.EdgeTable):
                # Edges may have been re-ordered, since sortedness requirements specify
                # they are sorted by parent time, and the relative order of
                # (unconnected) parent nodes might have changed due to time inference
                self.assertEquals(set(t1), set(t2))
            else:
                self.assertEquals(t1, t2)

    def test_simple_sim_1_tree(self):
        ts = msprime.simulate(8, mutation_rate=5, random_seed=2)
        dated_ts = tsdate.date(ts, Ne=1, mutation_rate=5)
        self.ts_equal_except_times(ts, dated_ts)

    def test_simple_sim_multi_tree(self):
        ts = msprime.simulate(8, mutation_rate=5, recombination_rate=5, random_seed=2)
        self.assertGreater(ts.num_trees, 1)
        dated_ts = tsdate.date(ts, Ne=1, mutation_rate=5)
        self.ts_equal_except_times(ts, dated_ts)

    def test_simple_sim_larger_example(self):
        # This makes ~1700 trees, and previously caused a failure
        ts = msprime.simulate(
            sample_size=10, length=2e6, Ne=10000, mutation_rate=1e-8,
            recombination_rate=1e-8, random_seed=11)
        dated_ts = tsdate.date(ts, Ne=10000, mutation_rate=1e-8)
        self.ts_equal_except_times(ts, dated_ts)

    def test_with_unary(self):
        ts = utility_functions.single_tree_ts_with_unary()
        dated_ts = tsdate.date(ts, Ne=1)
        self.ts_equal_except_times(ts, dated_ts)

    def test_non_contemporaneous(self):
        samples = [
            msprime.Sample(population=0, time=0),
            msprime.Sample(population=0, time=0),
            msprime.Sample(population=0, time=0),
            msprime.Sample(population=0, time=1.0)
        ]
        ts = msprime.simulate(samples=samples, Ne=1, mutation_rate=2)
        self.assertRaises(NotImplementedError, tsdate.date, ts, 1, 2)

    def test_truncated_ts(self):
        Ne = 1e2
        mu = 2e-4
        ts = msprime.simulate(
            10, Ne=Ne, length=400, recombination_rate=1e-4, mutation_rate=0,
            random_seed=12)
        truncated_ts = utility_functions.truncate_ts_samples(
            ts, average_span=200, random_seed=123)
        truncated_ts = msprime.mutate(truncated_ts, rate=mu, random_seed=12)
        dated_ts = tsdate.date(truncated_ts, Ne=Ne, mutation_rate=mu)
        # We should ideally test whether *haplotypes* are the same here
        # in case allele encoding has changed. But haplotypes() doesn't currently
        # deal with missing data
        self.ts_equal_except_times(truncated_ts, dated_ts)


class TestInferred(unittest.TestCase):
    """
    Tests for tsdate on simulated then inferred tree sequences.
    """
    def test_simple_sim_1_tree(self):
        ts = msprime.simulate(8, mutation_rate=5, random_seed=2)
        for use_times in [True, False]:
            sample_data = tsinfer.SampleData.from_tree_sequence(ts, use_times=use_times)
            inferred_ts = tsinfer.infer(sample_data)
            dated_ts = tsdate.date(inferred_ts, Ne=1, mutation_rate=5)
            self.assertTrue(
                all([a == b for a, b in zip(ts.haplotypes(), dated_ts.haplotypes())]))

    def test_simple_sim_multi_tree(self):
        ts = msprime.simulate(8, mutation_rate=5, recombination_rate=5, random_seed=2)
        self.assertGreater(ts.num_trees, 1)
        for use_times in [True, False]:
            sample_data = tsinfer.SampleData.from_tree_sequence(ts, use_times=use_times)
            inferred_ts = tsinfer.infer(sample_data)
            dated_ts = tsdate.date(inferred_ts, Ne=1, mutation_rate=5)
            self.assertTrue(
                all([a == b for a, b in zip(ts.haplotypes(), dated_ts.haplotypes())]))
