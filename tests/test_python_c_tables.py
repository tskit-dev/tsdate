"""
Low-level tests of the phasing, util and prior functions of the ``_tsdate`` module:
argument checking and the error paths that the oracle tests do not reach.
"""

import _tsdate
import numpy as np
import pytest
from oracle.helpers import b1, f64, i32, make_ts

# nodes 0, 1, 2 belong to individual 0; node 3 is the root
THREE_HAPLOTYPES = [(1, 0, 0), (1, 0, 0), (1, 0, 0), (0, 1, -1)]
STAR = [(0, 10, 3, 0), (0, 10, 3, 1), (0, 10, 3, 2)]


class TestBlockSingletons:
    def test_individual_with_three_edges(self):
        ts = make_ts(THREE_HAPLOTYPES, STAR, [(1, 0)], 10.0, 1)
        with pytest.raises(_tsdate.LibraryError, match="unphased individual"):
            _tsdate.block_singletons(
                b1(1),
                ts.nodes_individual,
                ts.mutations_node,
                ts.sites_position[ts.mutations_site],
                ts.edges_parent,
                ts.edges_child,
                ts.edges_left,
                ts.edges_right,
                ts.indexes_edge_insertion_order,
                ts.indexes_edge_removal_order,
                ts.sequence_length,
            )

    def test_wrong_dtype(self):
        ts = make_ts(THREE_HAPLOTYPES, STAR, [(1, 0)], 10.0, 1)
        with pytest.raises(TypeError):
            _tsdate.block_singletons(
                b1(1),
                ts.nodes_individual,
                ts.mutations_node.astype(np.int64),
                ts.sites_position[ts.mutations_site],
                ts.edges_parent,
                ts.edges_child,
                ts.edges_left,
                ts.edges_right,
                ts.indexes_edge_insertion_order,
                ts.indexes_edge_removal_order,
                ts.sequence_length,
            )


class TestReallocateUnphased:
    @staticmethod
    def call(likelihood, phase, mutations_block, blocks_edges):
        _tsdate.reallocate_unphased(
            likelihood,
            f64(*phase),
            i32(*mutations_block),
            np.array(blocks_edges, dtype=np.int32).reshape(-1, 2),
        )

    def test_block_out_of_bounds(self):
        with pytest.raises(_tsdate.LibraryError, match="out of bounds"):
            self.call(np.ones((2, 2)), [0.5], [1], [[0, 1]])

    def test_edge_out_of_bounds(self):
        with pytest.raises(_tsdate.LibraryError, match="out of bounds"):
            self.call(np.ones((2, 2)), [0.5], [0], [[0, 7]])

    def test_columns(self):
        with pytest.raises(ValueError, match="two columns"):
            self.call(np.ones((2, 3)), [0.5], [0], [[0, 1]])

    @pytest.mark.parametrize("kind", ["float32", "strided", "readonly"])
    def test_array_updated_in_place_must_not_be_converted(self, kind):
        # a copy would silently lose the update
        if kind == "float32":
            likelihood = np.ones((2, 2), dtype=np.float32)
        elif kind == "strided":
            likelihood = np.ones((2, 4))[:, ::2]
        else:
            likelihood = np.ones((2, 2))
            likelihood.flags.writeable = False
        with pytest.raises(ValueError):
            self.call(likelihood, [0.5], [0], [[0, 1]])


class TestMutationFrequency:
    def test_nodes_sample_dtype(self):
        ts = make_ts(THREE_HAPLOTYPES, STAR, [(1, 0)], 10.0, 1)
        with pytest.raises(TypeError):
            _tsdate.mutation_frequency(
                np.ones((ts.num_nodes, 1), dtype=np.float64),
                ts.mutations_node,
                ts.sites_position[ts.mutations_site],
                ts.edges_parent,
                ts.edges_child,
                ts.edges_left,
                ts.edges_right,
                ts.indexes_edge_insertion_order,
                ts.indexes_edge_removal_order,
                ts.sequence_length,
            )


class TestUtil:
    def test_constrain_ages_wrong_dtype(self):
        with pytest.raises(TypeError):
            _tsdate.constrain_ages(f64(0, 1), b1(1, 0), i32(1), np.array([0]), 0.1, 3)

    def test_contains_unary_nodes_mask_size(self):
        ts = make_ts(THREE_HAPLOTYPES, STAR, [], 10.0, 1)
        with pytest.raises(ValueError, match="number of nodes"):
            _tsdate.contains_unary_nodes(
                b1(0, 0),
                ts.edges_parent,
                ts.edges_left,
                ts.edges_right,
                ts.indexes_edge_insertion_order,
                ts.indexes_edge_removal_order,
                ts.sequence_length,
                ts.num_nodes,
            )


class TestPrior:
    def test_marginalize_over_ancestors_one_tip(self):
        with pytest.raises(_tsdate.LibraryError, match="Too few tips"):
            _tsdate.marginalize_over_ancestors(np.ones((1, 2)))

    def test_variance_no_tips(self):
        with pytest.raises(_tsdate.LibraryError, match="Too few tips"):
            _tsdate.conditional_coalescent_variance(0)

    def test_variance_negative_tips(self):
        with pytest.raises(ValueError, match="non-negative"):
            _tsdate.conditional_coalescent_variance(-1)

    def test_variance_one_tip(self):
        # the result is unused by tsdate
        assert _tsdate.conditional_coalescent_variance(1).tolist() == [0.0, 0.0]
