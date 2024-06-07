# MIT License
#
# Copyright (c) 2024 Tskit Developers
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
Test cases for tsdate utility functions
"""
import json
import logging

import msprime
import numpy as np
import pytest
import tsinfer
import tskit

import tsdate


class TestSplitDisjointNodes:
    @staticmethod
    def has_disjoint_nodes(ts):
        """
        Brute force check for disjoint nodes, by pulling out edge intervals for
        each node; taking the union of intervals; checking that a single
        interval remains
        """

        def merge_intervals(intervals):
            intervals = sorted(intervals, key=lambda x: x[0])
            result = []
            (start_candidate, stop_candidate) = intervals[0]
            for start, stop in intervals[1:]:
                if start <= stop_candidate:
                    stop_candidate = max(stop, stop_candidate)
                else:
                    result.append((start_candidate, stop_candidate))
                    (start_candidate, stop_candidate) = (start, stop)
            result.append((start_candidate, stop_candidate))
            return result

        intervals_by_node = {i: [] for i in range(ts.num_nodes)}
        for e in ts.edges():
            intervals_by_node[e.parent].append([e.left, e.right])
            intervals_by_node[e.child].append([e.left, e.right])

        for n in range(ts.num_nodes):
            intr = merge_intervals(intervals_by_node[n])
            if len(intr) != 1:
                return True

        return False

    @staticmethod
    def childset_changes_with_root(ts):
        """
        If root nodes are split whenever their children change, the next root
        should have the same child set if it has the same ID
        """
        last_childset = frozenset()
        last_root = tskit.NULL
        for t in ts.trees():
            if t.num_edges == 0:
                last_childset = frozenset()
                last_root = tskit.NULL
            else:
                if t.num_roots > 1:
                    return False
                childset = frozenset(list(t.children(t.root)))
                if t.root == last_root and childset != last_childset:
                    return False
                last_childset = childset
                last_root = t.root
        return True

    def test_nosplit(self):
        ts = tskit.Tree.generate_comb(5).tree_sequence
        split_ts = tsdate.util.split_disjoint_nodes(ts)
        assert ts.equals(split_ts, ignore_provenance=True)
        prov = json.loads(split_ts.provenance(-1).record)
        assert prov["software"]["name"] == "tsdate"
        assert prov["parameters"]["command"] == "split_disjoint_nodes"

    def test_simple(self, caplog):
        tables = tskit.Tree.generate_comb(5).tree_sequence.dump_tables()
        tables.delete_intervals([[0.2, 0.8]])
        tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
        ts = tables.tree_sequence()
        num_internal_nodes = ts.num_nodes - ts.num_samples
        with caplog.at_level(logging.WARNING):
            split_ts = tsdate.util.split_disjoint_nodes(ts)
            assert caplog.text == ""
        num_new_internal_nodes = split_ts.num_nodes - split_ts.num_samples
        assert split_ts.num_nodes > ts.num_nodes
        # all internal nodes should be split

        assert num_new_internal_nodes == num_internal_nodes * 2
        for node in split_ts.nodes():
            if node.is_sample():
                assert node.flags & tsdate.NODE_SPLIT_BY_PREPROCESS == 0
            else:
                assert node.flags & tsdate.NODE_SPLIT_BY_PREPROCESS != 0

    def test_metadata_warning(self, caplog):
        # Only sets extra metadata if schema is compatible
        ts = tskit.Tree.generate_comb(5).tree_sequence
        tables = ts.dump_tables()
        tables.delete_intervals([[0.2, 0.8]])
        tables.nodes.metadata_schema = tskit.MetadataSchema(
            {
                "codec": "struct",
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            }
        )
        ts = tables.tree_sequence()
        with caplog.at_level(logging.WARNING):
            tsdate.util.split_disjoint_nodes(ts)
            assert "Could not set 'unsplit_node_id'" in caplog.text

        tables.nodes.metadata_schema = tskit.MetadataSchema(None)
        tables.nodes.packset_metadata([b"xxx"] * ts.num_nodes)
        ts = tables.tree_sequence()
        tsdate.util.split_disjoint_nodes(ts)
        assert "Could not set 'unsplit_node_id'" in caplog.text
        for node in ts.nodes():
            assert node.metadata == b"xxx"

    def test_metadata(self):
        ts = tskit.Tree.generate_comb(5).tree_sequence
        tables = ts.dump_tables()
        tables.delete_intervals([[0.2, 0.8]])
        tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
        tables.nodes.packset_metadata(
            [
                tables.nodes.metadata_schema.validate_and_encode_row(
                    {"xxx": f"test{x}"}
                )
                for x in range(ts.num_nodes)
            ]
        )
        tables.nodes.flags = tables.nodes.flags | 1 << 16
        ts = tables.tree_sequence()
        split_ts = tsdate.util.split_disjoint_nodes(ts)
        is_nonsample = np.ones(split_ts.num_nodes, dtype=bool)
        is_nonsample[split_ts.samples()] = False
        _, counts = np.unique(split_ts.nodes_time[is_nonsample], return_counts=True)
        assert np.all(counts == 2)
        ids = {node.id: 0 for node in ts.nodes() if not node.is_sample()}
        for node in split_ts.nodes():
            if not node.is_sample():
                assert "unsplit_node_id" in node.metadata
                orig_node = ts.node(node.metadata["unsplit_node_id"])
                assert "unsplit_node_id" not in orig_node.metadata
                assert "xxx" in node.metadata
                assert "xxx" in orig_node.metadata
                assert node.metadata["xxx"] == orig_node.metadata["xxx"]
                assert node.time == orig_node.time
                assert node.flags == orig_node.flags | tsdate.NODE_SPLIT_BY_PREPROCESS
                ids[orig_node.id] += 1
        assert all([v == 2 for v in ids.values()])

    def test_no_provenance(self):
        ts = tskit.Tree.generate_comb(5).tree_sequence
        split_ts = tsdate.util.split_disjoint_nodes(ts, record_provenance=False)
        assert split_ts.num_provenances == ts.num_provenances
        split_ts = tsdate.util.split_disjoint_nodes(ts, record_provenance=True)
        assert split_ts.num_provenances == ts.num_provenances + 1

    def test_inferred(self):
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
        split_ts = tsdate.util.split_disjoint_nodes(inferred_ts)
        assert self.has_disjoint_nodes(inferred_ts)
        assert not self.has_disjoint_nodes(split_ts)
        assert split_ts.num_edges == inferred_ts.num_edges
        assert split_ts.num_nodes > inferred_ts.num_nodes


class TestPreprocessTs:
    def test_no_sites(self):
        ts = tskit.Tree.generate_comb(3).tree_sequence
        with pytest.raises(ValueError, match="no sites"):
            ts = tsdate.preprocess_ts(ts)

    def test_split_disjoint(self):
        tables = tskit.Tree.generate_comb(5).tree_sequence.dump_tables()
        tables.delete_intervals([[0.2, 0.8]])
        tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
        tables.sites.add_row(0.1, "A")
        ts = tables.tree_sequence()
        num_nonsample_nodes = ts.num_nodes - ts.num_samples
        ts = tsdate.preprocess_ts(ts)
        num_split_nonsample_nodes = ts.num_nodes - ts.num_samples
        assert num_split_nonsample_nodes == 2 * num_nonsample_nodes

    def test_no_split_disjoint(self):
        tables = tskit.Tree.generate_comb(5).tree_sequence.dump_tables()
        tables.delete_intervals([[0.2, 0.8]])
        tables.sites.add_row(0.1, "A")
        ts = tables.tree_sequence()
        num_nodes = ts.num_nodes
        ts = tsdate.preprocess_ts(ts, split_disjoint=False)
        assert ts.num_nodes == num_nodes

    def test_is_simplified(self):
        tables = tskit.Tree.generate_comb(5).tree_sequence.dump_tables()
        tables.simplify(np.arange(4), keep_unary=True)  # leaves a unary node
        tables.sites.add_row(0.5, "A")
        tables.populations.add_row()
        tables.individuals.add_row()
        ts = tables.tree_sequence()
        tree = ts.first()
        # Check there is a single unary node
        assert sum(tree.num_children(u) == 1 for u in tree.nodes()) == 1
        num_nodes = ts.num_nodes
        num_populations = ts.num_populations
        num_sites = ts.num_sites
        num_individuals = ts.num_individuals
        ts = tsdate.preprocess_ts(ts)
        assert ts.num_nodes == num_nodes - 1  # Unary node removed
        assert ts.num_populations == num_populations
        assert ts.num_sites == num_sites
        assert ts.num_individuals == num_individuals

    def test_simplified_params_passed(self):
        tables = tskit.Tree.generate_comb(3).tree_sequence.dump_tables()
        tables.sites.add_row(0.5, "A")
        tables.populations.add_row()
        tables.individuals.add_row()
        ts = tables.tree_sequence()
        num_populations = ts.num_populations
        num_individuals = ts.num_individuals
        ts = tsdate.preprocess_ts(ts, filter_individuals=True)
        assert ts.num_populations == num_populations
        assert ts.num_individuals == num_individuals - 1

    def test_record_provenance(self):
        tables = tskit.Tree.generate_comb(3).tree_sequence.dump_tables()
        tables.sites.add_row(0.5, "A")
        ts = tables.tree_sequence()
        num_provenances = ts.num_provenances
        ts = tsdate.preprocess_ts(ts)
        assert ts.num_provenances == num_provenances + 1
        prov = json.loads(ts.provenance(-1).record)
        assert prov["software"]["name"] == "tsdate"
        assert prov["parameters"]["command"] == "preprocess_ts"
        ts = tsdate.preprocess_ts(ts, record_provenance=False)
        assert ts.num_provenances == num_provenances + 1

    def test_trim_flanks(self):
        tables = tskit.Tree.generate_comb(3, span=100).tree_sequence.dump_tables()
        tables.sites.add_row(10, "A")
        tables.sites.add_row(90, "A")
        ts = tables.tree_sequence()
        assert ts.sequence_length == 100
        assert ts.num_trees == 1
        ts = tsdate.preprocess_ts(ts)
        assert ts.num_trees == 3
        assert ts.first().num_edges == 0
        assert ts.first().interval.right == 10 - 1
        assert ts.last().num_edges == 0
        assert ts.last().interval.left == 90 + 1

    def test_sim_example(self):
        # Test a larger example
        ts = msprime.sim_ancestry(
            20,
            sequence_length=1e4,
            recombination_rate=0.0005,
            record_full_arg=True,
            random_seed=1,
        )
        tables = msprime.sim_mutations(ts, rate=0.01, random_seed=1).dump_tables()
        tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
        ts = tables.tree_sequence()
        num_nodes = ts.simplify().num_nodes
        num_trees = ts.simplify().num_trees
        assert num_trees > 50
        ts = tsdate.preprocess_ts(ts)
        assert ts.num_nodes > num_nodes  # Nodes added by split_disjoint
        assert np.sum((ts.nodes_flags & tsdate.NODE_SPLIT_BY_PREPROCESS) != 0) > 0
        first_empty = int(ts.first().num_edges == 0)
        last_empty = int(ts.last().num_edges == 0)
        # Next assumes no breakpoints before first site or after last
        assert ts.num_trees == num_trees + first_empty + last_empty

    # TODO - test minimum_gap param


class TestUnaryNodeCheck:
    def test_inferred(self):
        ts = msprime.sim_ancestry(
            10,
            population_size=1e4,
            recombination_rate=1e-8,
            sequence_length=1e6,
            random_seed=1,
        )
        ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=1)
        sample_data = tsinfer.SampleData.from_tree_sequence(ts)
        inferred_ts = tsinfer.infer(sample_data)
        simplified_ts = inferred_ts.simplify()
        assert tsdate.util.contains_unary_nodes(inferred_ts)
        assert not tsdate.util.contains_unary_nodes(simplified_ts)
        with pytest.raises(ValueError, match="contains unary nodes"):
            tsdate.date(inferred_ts, mutation_rate=1e-8, method="variational_gamma")

    def test_simulated(self):
        ts = msprime.sim_ancestry(
            10,
            population_size=1e4,
            recombination_rate=1e-8,
            sequence_length=1e6,
            random_seed=1,
            record_full_arg=True,
        )
        ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=1)
        simplified_ts = ts.simplify()
        assert tsdate.util.contains_unary_nodes(ts)
        assert not tsdate.util.contains_unary_nodes(simplified_ts)
        with pytest.raises(ValueError, match="contains unary nodes"):
            tsdate.date(ts, mutation_rate=1e-8, method="variational_gamma")
