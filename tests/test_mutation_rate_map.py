# MIT License
#
# Copyright (c) 2026 Tskit developers
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
Test mutation rate map support
"""

from string import ascii_lowercase, ascii_uppercase

import numpy as np
import pytest
import tskit

import tsdate


def example_transform_pair():
    """
        original:

        7.00┊         ┊         ┊    10   ┊         ┊
            ┊         ┊         ┊   ┏━┻━┓ ┊         ┊
        6.00┊         ┊         ┊   ┃   ┃ ┊     9   ┊
            ┊         ┊         ┊   ┃   ┃ ┊   ┏━┻━┓ ┊
        5.00┊     8   ┊     8   ┊   ┃   ┃ ┊   ┃   ┃ ┊
            ┊   ┏━┻━┓ ┊   ┏━┻━┓ ┊   ┃   ┃ ┊   ┃   ┃ ┊
        4.00┊   7   ┃ ┊   7   ┃ ┊   7   ┃ ┊   7   ┃ ┊
            ┊  ┏┻━┓ ┃ ┊  ┏┻━┓ ┃ ┊  ┏┻━┓ ┃ ┊  ┏┻━┓ ┃ ┊
        3.00┊  ┃  ┃ ┃ ┊  ┃  ┃ ┃ ┊  6  ┃ ┃ ┊  6  ┃ ┃ ┊
            ┊  ┃  ┃ ┃ ┊  ┃  ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┊
        2.00┊  ┃  ┃ ┃ ┊  5  ┃ ┃ ┊ ┃ ┃ ┃ ┃ ┊ ┃ ┃ ┃ ┃ ┊
            ┊  ┃  ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┊ ┃ ┃ ┃ ┃ ┊ ┃ ┃ ┃ ┃ ┊
        1.00┊  4  ┃ ┃ ┊ ┃ ┃ ┃ ┃ ┊ ┃ ┃ ┃ ┃ ┊ ┃ ┃ ┃ ┃ ┊
            ┊ ┏┻┓ ┃ ┃ ┊ ┃ ┃ ┃ ┃ ┊ ┃ ┃ ┃ ┃ ┊ ┃ ┃ ┃ ┃ ┊
        0.00┊ 0 2 1 3 ┊ 0 2 1 3 ┊ 0 2 1 3 ┊ 0 2 1 3 ┊
            0        25        50        75        100

    transformed to:

    7.00┊         ┊         ┊
        ┊         ┊         ┊
    6.00┊         ┊     9   ┊
        ┊         ┊   ┏━┻━┓ ┊
    5.00┊     8   ┊   ┃   ┃ ┊
        ┊   ┏━┻━┓ ┊   ┃   ┃ ┊
    4.00┊   7   ┃ ┊   7   ┃ ┊
        ┊  ┏┻━┓ ┃ ┊  ┏┻━┓ ┃ ┊
    3.00┊  ┃  ┃ ┃ ┊  6  ┃ ┃ ┊
        ┊  ┃  ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┊
    2.00┊  ┃  ┃ ┃ ┊ ┃ ┃ ┃ ┃ ┊
        ┊  ┃  ┃ ┃ ┊ ┃ ┃ ┃ ┃ ┊
    1.00┊  4  ┃ ┃ ┊ ┃ ┃ ┃ ┃ ┊
        ┊ ┏┻┓ ┃ ┃ ┊ ┃ ┃ ┃ ┃ ┊
    0.00┊ 0 2 1 3 ┊ 0 2 1 3 ┊
        0         2        52
    """
    # original space
    tab = tskit.TableCollection()
    tab.nodes.set_columns(
        flags=[tskit.NODE_IS_SAMPLE] * 4 + [0] * 7,
        time=[0] * 4 + list(range(1, 8)),
    )
    tab.edges.set_columns(
        parent=[4, 4, 5, 5, 6, 6, 7, 7, 7, 7, 8, 8, 9, 9, 10, 10],
        child=[0, 2, 0, 2, 0, 2, 1, 4, 5, 6, 3, 7, 3, 7, 3, 7],
        left=[0, 0, 25, 25, 50, 50, 0, 0, 25, 50, 0, 0, 75, 75, 50, 50],
        right=[25, 25, 50, 50, 100, 100, 100, 25, 50, 100, 50, 50, 100, 100, 75, 75],
    )
    site_id = list(range(11))
    ancestral_state = [str(x).encode("ascii") for x in site_id]
    ancestral_state, ancestral_state_offset = tskit.pack_bytes(ancestral_state)
    site_metadata = [ascii_uppercase[i].encode("ascii") for i in site_id]
    site_metadata, site_metadata_offset = tskit.pack_bytes(site_metadata)
    tab.sites.set_columns(
        position=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99],
        ancestral_state=ancestral_state,
        ancestral_state_offset=ancestral_state_offset,
        metadata=site_metadata,
        metadata_offset=site_metadata_offset,
    )
    mut_id = list(range(16))
    derived_state = [str(x).encode("ascii") for x in mut_id]
    derived_state, derived_state_offset = tskit.pack_bytes(derived_state)
    mut_metadata = [ascii_lowercase[i].encode("ascii") for i in mut_id]
    mut_metadata, mut_metadata_offset = tskit.pack_bytes(mut_metadata)
    tab.mutations.set_columns(
        site=[0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9, 10],
        parent=[-1, -1, 1, -1, -1, 4, -1, -1, 7, -1, -1, 10, -1, -1, 13, -1],
        node=[7] * 16,
        derived_state=derived_state,
        derived_state_offset=derived_state_offset,
        metadata=mut_metadata,
        metadata_offset=mut_metadata_offset,
    )
    tab.sequence_length = 100
    ts = tab.tree_sequence()
    # transform
    ratemap = tskit.RateMap(rate=[0.1, 0.0, 2.0], position=[0, 20, 75, 100])
    # transformed space
    trans_tab = tskit.TableCollection()
    trans_tab.nodes.set_columns(
        flags=[tskit.NODE_IS_SAMPLE] * 4 + [0] * 7,
        time=[0] * 4 + list(range(1, 8)),
    )
    trans_tab.edges.set_columns(
        parent=[4, 4, 6, 6, 7, 7, 7, 8, 8, 9, 9],
        child=[0, 2, 0, 2, 1, 4, 6, 3, 7, 3, 7],
        left=[0, 0, 2, 2, 0, 0, 2, 0, 0, 2, 2],
        right=[2, 2, 52, 52, 52, 2, 52, 2, 2, 52, 52],
    )
    site_subset = [0, 1, 8, 9, 10]
    ancestral_state = [str(x).encode("ascii") for x in site_subset]
    ancestral_state, ancestral_state_offset = tskit.pack_bytes(ancestral_state)
    site_metadata = [ascii_uppercase[i].encode("ascii") for i in site_subset]
    site_metadata, site_metadata_offset = tskit.pack_bytes(site_metadata)
    trans_tab.sites.set_columns(
        position=[0, 1, 12, 32, 50],
        ancestral_state=ancestral_state,
        ancestral_state_offset=ancestral_state_offset,
        metadata=site_metadata,
        metadata_offset=site_metadata_offset,
    )
    mut_subset = [0, 1, 2, 12, 13, 14, 15]
    derived_state = [str(x).encode("ascii") for x in mut_subset]
    derived_state, derived_state_offset = tskit.pack_bytes(derived_state)
    mut_metadata = [ascii_lowercase[i].encode("ascii") for i in mut_subset]
    mut_metadata, mut_metadata_offset = tskit.pack_bytes(mut_metadata)
    trans_tab.mutations.set_columns(
        site=[0, 1, 1, 2, 3, 3, 4],
        parent=[-1, -1, 1, -1, -1, 4, -1],
        node=[7] * 7,
        derived_state=derived_state,
        derived_state_offset=derived_state_offset,
        metadata=mut_metadata,
        metadata_offset=mut_metadata_offset,
    )
    trans_tab.sequence_length = 52
    trans_ts = trans_tab.tree_sequence()
    return ts, trans_ts, ratemap


def test_transform_coordinates_by_ratemap():
    """
    Test that transform produces expected result
    """
    ts, trans_ts, ratemap = example_transform_pair()
    trans_ts_ck = tsdate.util.transform_coordinates_by_ratemap(ts, ratemap)
    assert trans_ts_ck == trans_ts


def test_transform_coordinates_identity():
    """
    When the ratemap rates are all one, the tree sequence should be unmodified
    """
    ts, _, ratemap = example_transform_pair()
    ratemap = tskit.RateMap(rate=np.ones_like(ratemap.rate), position=ratemap.position)
    trans_ts = tsdate.util.transform_coordinates_by_ratemap(ts, ratemap)
    assert ts == trans_ts


def test_transform_coordinates_nil():
    """
    When the ratemap rates are all zero, the tree sequence is empty
    """
    ts, _, ratemap = example_transform_pair()
    ratemap = tskit.RateMap(rate=np.zeros_like(ratemap.rate), position=ratemap.position)
    with pytest.raises(tskit.LibraryError, match="Sequence length must be > 0"):
        tsdate.util.transform_coordinates_by_ratemap(ts, ratemap)


def test_transform_coordinates_nan():
    """
    NaNs are treated like zeros: they do not contribute to cumulative mass,
    and sites in intervals with NaN rate are removed because np.nan > 0 is False.
    """
    ts, _, ratemap = example_transform_pair()
    ratemap_nan = tskit.RateMap(
        rate=np.append(np.append(np.nan, ratemap.rate), np.nan),
        position=np.concatenate([[0, 10], ratemap.position[1:-1], [90, 100]]),
    )
    ratemap_ck = tskit.RateMap(
        rate=np.append(np.append(0.0, ratemap.rate), 0.0),
        position=np.concatenate([[0, 10], ratemap.position[1:-1], [90, 100]]),
    )
    trans_ts_nan = tsdate.util.transform_coordinates_by_ratemap(ts, ratemap_nan)
    trans_ts_ck = tsdate.util.transform_coordinates_by_ratemap(ts, ratemap_ck)
    assert trans_ts_ck == trans_ts_nan
