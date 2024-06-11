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

import numpy as np
import pytest
import tskit
import msprime
import tsinfer

from tsdate.phasing import block_singletons
from tsdate.phasing import count_mutations


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
    return inferred_ts


class TestCountMutations:
    def test_count_mutations(self, inferred_ts):
        edge_stats, muts_edge = count_mutations(inferred_ts)
        ck_edge_muts = np.zeros(inferred_ts.num_edges)
        ck_muts_edge = np.full(inferred_ts.num_mutations, tskit.NULL)
        for m in inferred_ts.mutations():
            if m.edge != tskit.NULL:
                ck_edge_muts[m.edge] += 1.0
                ck_muts_edge[m.id] = m.edge
        ck_edge_span = inferred_ts.edges_right - inferred_ts.edges_left
        np.testing.assert_array_almost_equal(ck_edge_muts, edge_stats[:, 0])
        np.testing.assert_array_almost_equal(ck_edge_span, edge_stats[:, 1])
        np.testing.assert_array_equal(ck_muts_edge, muts_edge)


class TestBlockSingletons:

    @staticmethod
    def naive_block_singletons(ts, individual):
        """
        Get all intervals where the two intermediate parents of an individual are
        unchanged over the interval.
        """
        i = individual
        j, k = ts.individual(i).nodes
        last_block = np.full(2, tskit.NULL)
        last_span = np.zeros(2)
        muts_edges = np.full((ts.num_mutations, 2), tskit.NULL)
        blocks_edge = []
        blocks_span = []
        for tree in ts.trees():
            if tree.num_edges == 0: # skip tree
                muts = []
                span = 0.0
                block = tskit.NULL, tskit.NULL
            else:
                muts = [m.id for m in tree.mutations() if m.node == j or m.node == k]
                span = tree.interval.span
                block = tree.edge(j), tree.edge(k)
                for m in muts:
                    muts_edges[m] = block
            if last_block[0] != tskit.NULL and not np.array_equal(block, last_block): # flush block
                blocks_edge.extend(last_block)
                blocks_span.extend(last_span)
                last_span[:] = 0.0
            last_span += len(muts), span
            last_block[:] = block
        if last_block[0] != tskit.NULL: # flush last block
            blocks_edge.extend(last_block)
            blocks_span.extend(last_span)
        blocks_edge = np.array(blocks_edge).reshape(-1, 2)
        blocks_span = np.array(blocks_span).reshape(-1, 2)
        total_span = np.sum([t.interval.span for t in ts.trees() if t.num_edges > 0])
        total_muts = np.sum(np.logical_or(ts.mutations_node == j, ts.mutations_node == k))
        assert np.sum(blocks_span[:, 0]) == total_muts
        assert np.sum(blocks_span[:, 1]) == total_span
        return blocks_span, blocks_edge, muts_edges

    def test_against_naive(self, inferred_ts):
        """
        Test fast routine against simpler tree-by-tree,
        individual-by-individual implementation
        """
        ts = inferred_ts
        individuals_unphased = np.full(ts.num_individuals, False)
        unphased_individuals = np.arange(0, ts.num_individuals // 2)
        individuals_unphased[unphased_individuals] = True
        block_stats, block_edges, muts_block = block_singletons(ts, individuals_unphased)
        block_edges = block_edges
        singletons = muts_block != tskit.NULL
        muts_edges = np.full((ts.num_mutations, 2), tskit.NULL)
        muts_edges[singletons] = block_edges[muts_block[singletons]]
        ck_num_blocks = 0
        ck_num_singletons = 0
        for i in np.flatnonzero(individuals_unphased):
            ck_block_stats, ck_block_edges, ck_muts_edges = self.naive_block_singletons(ts, i)
            ck_num_blocks += ck_block_stats.shape[0]
            # blocks of individual i
            nodes_i = ts.individual(i).nodes
            blocks_i = np.isin(ts.edges_child[block_edges.min(axis=1)], nodes_i)
            np.testing.assert_allclose(block_stats[blocks_i], ck_block_stats)
            np.testing.assert_array_equal(
                np.min(block_edges[blocks_i], axis=1), np.min(ck_block_edges, axis=1)
            )
            np.testing.assert_array_equal(
                np.max(block_edges[blocks_i], axis=1), np.max(ck_block_edges, axis=1)
            )
            # singleton mutations in unphased individual i
            ck_muts_i = ck_muts_edges[:, 0] != tskit.NULL
            np.testing.assert_array_equal(
                np.min(muts_edges[ck_muts_i], axis=1), 
                np.min(ck_muts_edges[ck_muts_i], axis=1),
            )
            np.testing.assert_array_equal(
                np.max(muts_edges[ck_muts_i], axis=1), 
                np.max(ck_muts_edges[ck_muts_i], axis=1),
            )
            ck_num_singletons += np.sum(ck_muts_i)
        assert ck_num_blocks == block_stats.shape[0] == block_edges.shape[0]
        assert ck_num_singletons == np.sum(singletons)

    def test_total_counts(self, inferred_ts):
        """
        Sanity check: total number of mutations should equal number of singletons
        and total edge span should equal sum of spans of singleton edges
        """
        ts = inferred_ts
        individuals_unphased = np.full(ts.num_individuals, False)
        unphased_individuals = np.arange(0, ts.num_individuals // 2)
        individuals_unphased[unphased_individuals] = True
        unphased_nodes = np.concatenate([ts.individual(i).nodes for i in unphased_individuals])
        total_singleton_span = 0.0
        total_singleton_muts = 0.0
        for t in ts.trees():
            if t.num_edges == 0: continue
            for s in t.samples():
                if s in unphased_nodes:
                    total_singleton_span += t.span
            for m in t.mutations():
                if t.num_samples(m.node) == 1 and (m.node in unphased_nodes):
                    e = t.edge(m.node)
                    total_singleton_muts += 1.0
        block_stats, *_ = block_singletons(ts, individuals_unphased)
        assert np.isclose(np.sum(block_stats[:, 0]), total_singleton_muts)
        assert np.isclose(np.sum(block_stats[:, 1]), total_singleton_span / 2)

    def test_singleton_edges(self, inferred_ts):
        """
        Sanity check: all singleton edges attached to unphased individuals
        should show up in blocks
        """
        ts = inferred_ts
        individuals_unphased = np.full(ts.num_individuals, False)
        unphased_individuals = np.arange(0, ts.num_individuals // 2)
        individuals_unphased[unphased_individuals] = True
        unphased_nodes = set(np.concatenate([ts.individual(i).nodes for i in unphased_individuals]))
        ck_singleton_edge = set()
        for t in ts.trees():
            if t.num_edges == 0: continue
            for s in ts.samples():
                if s in unphased_nodes:
                    ck_singleton_edge.add(t.edge(s))
        _, block_edges, *_ = block_singletons(ts, individuals_unphased)
        singleton_edge = set([i for i in block_edges.flatten()])
        assert singleton_edge == ck_singleton_edge

    def test_singleton_mutations(self, inferred_ts):
        """
        Sanity check: all singleton mutations in unphased individuals
        should show up in blocks
        """
        ts = inferred_ts
        individuals_unphased = np.full(ts.num_individuals, False)
        unphased_individuals = np.arange(0, ts.num_individuals // 2)
        individuals_unphased[unphased_individuals] = True
        unphased_nodes = np.concatenate([ts.individual(i).nodes for i in unphased_individuals])
        ck_singleton_muts = set()
        for t in ts.trees():
            if t.num_edges == 0: continue
            for m in t.mutations():
                if t.num_samples(m.node) == 1 and (m.node in unphased_nodes):
                    ck_singleton_muts.add(m.id)
        _, _, block_muts = block_singletons(ts, individuals_unphased)
        singleton_muts = set([i for i in np.flatnonzero(block_muts != tskit.NULL)])
        assert singleton_muts == ck_singleton_muts

    def test_all_phased(self, inferred_ts):
        """
        Test that empty arrays are returned when all individuals are phased
        """
        ts = inferred_ts
        individuals_unphased = np.full(ts.num_individuals, False)
        block_stats, block_edges, block_muts = block_singletons(ts, individuals_unphased)
        assert block_stats.shape == (0, 2)
        assert block_edges.shape == (0, 2)
        assert np.all(block_muts == tskit.NULL)

