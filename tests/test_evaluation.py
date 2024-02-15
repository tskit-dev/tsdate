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
Test tools for mapping between node sets of different tree sequences
"""
from collections import defaultdict
from itertools import combinations

import msprime
import numpy as np
import pytest
import scipy.sparse
import tsinfer

from tsdate import evaluation

# --- simulate test case ---
demo = msprime.Demography.isolated_model([1e4])
for t in np.linspace(500, 10000, 20):
    demo.add_census(time=t)
true_unary = msprime.sim_ancestry(
    samples=10,
    sequence_length=1e6,
    demography=demo,
    recombination_rate=1e-8,
    random_seed=1024,
)
true_unary = msprime.sim_mutations(true_unary, rate=2e-8, random_seed=1024)
assert true_unary.num_trees > 1
true_simpl = true_unary.simplify(filter_sites=False)
sample_dat = tsinfer.SampleData.from_tree_sequence(true_simpl)
infr_unary = tsinfer.infer(sample_dat)
infr_simpl = infr_unary.simplify(filter_sites=False)


def naive_shared_node_spans(ts, other):
    """
    Inefficient but transparent function to get span where nodes from two tree
    sequences subtend the same sample set
    """

    def _clade_dict(tree):
        clade_to_node = defaultdict(set)
        for node in tree.nodes():
            clade = frozenset(tree.samples(node))
            clade_to_node[clade].add(node)
        return clade_to_node

    assert ts.sequence_length == other.sequence_length
    assert ts.num_samples == other.num_samples
    out = np.zeros((ts.num_nodes, other.num_nodes))
    for interval, query_tree, target_tree in ts.coiterate(other):
        query = _clade_dict(query_tree)
        target = _clade_dict(target_tree)
        span = interval.right - interval.left
        for clade, nodes in query.items():
            if clade in target:
                for i in nodes:
                    for j in target[clade]:
                        out[i, j] += span
    return scipy.sparse.csr_matrix(out)


@pytest.mark.parametrize("ts", [true_unary, infr_unary, true_simpl, infr_simpl])
class TestCladeMap:
    def test_map(self, ts):
        """
        test that clade map has correct nodes, clades
        """
        clade_map = evaluation.CladeMap(ts)
        for tree in ts.trees():
            for node in tree.nodes():
                clade = frozenset(tree.samples(node))
                assert node in clade_map._nodes[clade]
                assert clade_map._clades[node] == clade
            clade_map.next()

    def test_diff(self, ts):
        """
        test difference in clades between adjacent trees
        """
        clade_map = evaluation.CladeMap(ts)
        tree_1 = ts.first()
        tree_2 = ts.first()
        while True:
            tree_2.next()
            diff = clade_map.next()
            diff_test = {}
            for n in set(tree_1.nodes()) | set(tree_2.nodes()):
                prev = frozenset(tree_1.samples(n))
                curr = frozenset(tree_2.samples(n))
                if prev != curr:
                    diff_test[n] = (prev, curr)
            for node in diff_test.keys() | diff.keys():
                assert diff_test[node][0] == diff[node][0]
                assert diff_test[node][1] == diff[node][1]
            if tree_2.index == ts.num_trees - 1:
                break
            tree_1.next()


class TestNodeMatching:
    @pytest.mark.parametrize(
        "pair", combinations([infr_simpl, true_simpl, infr_unary, true_unary], 2)
    )
    def test_shared_spans(self, pair):
        """
        Check that efficient implementation returns same answer as naive
        implementation
        """
        check = naive_shared_node_spans(pair[0], pair[1])
        test = evaluation.shared_node_spans(pair[0], pair[1])
        assert check.shape == test.shape
        assert check.nnz == test.nnz
        assert np.allclose(check.data, test.data)

    @pytest.mark.parametrize("ts", [infr_simpl, true_simpl])
    def test_match_self(self, ts):
        """
        Check that matching against self returns node ids

        TODO: this'll only work reliably when there's not unary nodes.
        """
        time, _, hit = evaluation.match_node_ages(ts, ts)
        assert np.allclose(time, ts.nodes_time)
        assert np.array_equal(hit, np.arange(ts.num_nodes))
