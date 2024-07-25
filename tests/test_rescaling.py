# MIT License
#
# Copyright (c) 2021-23 Tskit Developers
# Copyright (c) 2020-21 University of Oxford
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
Test cases for the gamma-variational approximations in tsdate
"""

from collections import defaultdict

import msprime
import numpy as np
import pytest
import tsinfer
import tskit

import tsdate
from tsdate.rescaling import count_mutations, mutational_area


@pytest.fixture(scope="session")
def inferred_ts():
    ts = msprime.sim_ancestry(
        10,
        population_size=1e4,
        recombination_rate=1e-8,
        sequence_length=1e6,
        random_seed=1,
    )
    ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=1)
    sample_data = tsinfer.SampleData.from_tree_sequence(ts)
    inferred_ts = tsinfer.infer(sample_data).simplify()
    inferred_ts = tsdate.date(inferred_ts, mutation_rate=1e-8, max_iterations=2)
    return inferred_ts


class TestMutationalArea:
    """
    Test tallying of mutational area within inter-node time intervals.
    """

    @staticmethod
    def naive_mutational_area(ts):
        """
        Count muts/area in each inter-node interval
        """
        edges_muts = np.zeros(ts.num_edges)
        for m in ts.mutations():
            if m.edge != tskit.NULL:
                edges_muts[m.edge] += 1.0
        unique_node_times, node_map = np.unique(ts.nodes_time, return_inverse=True)
        area = np.zeros(unique_node_times.size - 1)
        muts = np.zeros(unique_node_times.size - 1)
        for edge in ts.edges():
            p = node_map[edge.parent]
            c = node_map[edge.child]
            length = ts.nodes_time[edge.parent] - ts.nodes_time[edge.child]
            width = edge.right - edge.left
            area[c:p] += width
            muts[c:p] += edges_muts[edge.id] / length
        return muts, area, np.diff(unique_node_times), node_map

    @staticmethod
    def naive_total_path_area(ts):
        """
        Count total area of all paths from samples to roots,
        and all mutations on these paths.
        """
        muts, span = 0.0, 0.0
        for t in ts.trees():
            if t.num_edges == 0:
                continue
            mut_node = defaultdict(int)
            for m in t.mutations():
                mut_node[m.node] += 1
            for c in t.nodes():
                p = t.parent(c)
                if t.parent(c) == tskit.NULL:
                    continue
                weight = t.num_samples(c)
                length = t.time(p) - t.time(c)
                muts += mut_node[c] * weight
                span += t.span * length * weight
        return muts, span

    def test_total_mutational_area(self, inferred_ts):
        ts = inferred_ts
        likelihoods, _ = count_mutations(ts)
        epoch_muts, epoch_span, epoch_duration, _ = mutational_area(
            ts.nodes_time,
            likelihoods,
            ts.edges_parent,
            ts.edges_child,
        )
        segsite = np.sum(epoch_span * epoch_duration)
        ck_segsite = ts.segregating_sites(mode="branch", span_normalise=False)
        assert np.isclose(segsite, ck_segsite)
        totmuts = np.sum(epoch_muts * epoch_duration)
        ck_totmuts = ts.segregating_sites(mode="site", span_normalise=False)
        assert np.isclose(totmuts, ck_totmuts)

    def test_total_path_area(self, inferred_ts):
        ts = inferred_ts
        likelihoods, _ = tsdate.rescaling.count_mutations(ts, size_biased=True)
        epoch_muts, epoch_span, epoch_duration, _ = mutational_area(
            ts.nodes_time,
            likelihoods,
            ts.edges_parent,
            ts.edges_child,
        )
        totmuts = np.sum(epoch_muts * epoch_duration)
        patharea = np.sum(epoch_span * epoch_duration)
        ck_totmuts, ck_patharea = self.naive_total_path_area(ts)
        assert np.isclose(totmuts, ck_totmuts)
        assert np.isclose(patharea, ck_patharea)

    def test_vs_naive(self, inferred_ts):
        ts = inferred_ts
        likelihoods, _ = count_mutations(inferred_ts)
        epoch_muts, epoch_span, epoch_duration, node_index = mutational_area(
            ts.nodes_time,
            likelihoods,
            ts.edges_parent,
            ts.edges_child,
        )
        ck_muts, ck_span, ck_duration, ck_index = self.naive_mutational_area(ts)
        np.testing.assert_allclose(epoch_muts, ck_muts)
        np.testing.assert_allclose(epoch_span, ck_span)
        np.testing.assert_allclose(epoch_duration, ck_duration)
        np.testing.assert_allclose(node_index, ck_index)


class TestCountMutations:
    """
    Test tallying of mutations on edges
    """

    @staticmethod
    def naive_count_mutations(ts):
        edge_muts = np.zeros(ts.num_edges)
        muts_edge = np.full(ts.num_mutations, tskit.NULL)
        for m in ts.mutations():
            if m.edge != tskit.NULL:
                edge_muts[m.edge] += 1.0
                muts_edge[m.id] = m.edge
        edge_span = ts.edges_right - ts.edges_left
        return np.column_stack([edge_muts, edge_span]), muts_edge

    @staticmethod
    def naive_count_sizebiased(ts):
        muts_edge = np.full(ts.num_mutations, tskit.NULL)
        edge_muts = np.zeros(ts.num_edges)
        edge_span = np.zeros(ts.num_edges)
        for m in ts.mutations():
            if m.edge != tskit.NULL:
                muts_edge[m.id] = m.edge
        for t in ts.trees():
            if t.num_edges == 0:
                continue
            for m in t.mutations():
                e = t.edge(m.node)
                if e == tskit.NULL:
                    continue
                edge_muts[e] += t.num_samples(m.node)
            for n in t.nodes():
                e = t.edge(n)
                if e == tskit.NULL:
                    continue
                edge_span[e] += t.span * t.num_samples(n)
        return np.column_stack([edge_muts, edge_span]), muts_edge

    def test_count_mutations(self, inferred_ts):
        edge_stats, muts_edge = count_mutations(inferred_ts)
        ck_edge_stats, ck_muts_edge = self.naive_count_mutations(inferred_ts)
        np.testing.assert_array_almost_equal(ck_edge_stats, edge_stats)
        np.testing.assert_array_equal(ck_muts_edge, muts_edge)

    def test_count_sizebiased(self, inferred_ts):
        edge_stats, muts_edge = count_mutations(inferred_ts, size_biased=True)
        ck_edge_stats, ck_muts_edge = self.naive_count_sizebiased(inferred_ts)
        np.testing.assert_array_almost_equal(ck_edge_stats, edge_stats)
        np.testing.assert_array_equal(ck_muts_edge, muts_edge)

    @pytest.mark.skip("Ancient samples not implemented")
    def test_count_sizebiased_with_ancient(self, inferred_ts):
        # TODO: if there are ancestral samples, these should not be used as weights.
        # test when ancestral samples are fully implemented.
        return

    @pytest.mark.skip("Accessibility mask not implemented")
    def test_count_mutations_with_accessible(self, inferred_ts):
        # TODO
        return
