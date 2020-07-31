# MIT License
#
# Copyright (c) 2020 University of Oxford
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
Utility functions for tsdate. Many of these can be removed when tskit is updated to
a more recent version which has the functionality built-in
"""

import numpy as np
import tsinfer

def edge_span(edge):
    return edge.right - edge.left


def tree_num_children(tree, node):
    return len(tree.children(node))


def get_single_root(tree):
    # TODO - use new 'root_threshold=2' to avoid having to check isolated nodes
    topological_roots = [r for r in tree.roots if tree_num_children(tree, r) != 0]
    if len(topological_roots) > 1:
        raise ValueError(
            "Invalid tree sequence: tree {} has >1 root".format(tree.index))
    if len(topological_roots) == 0:
        return None  # Empty tree
    return topological_roots[0]


def reduce_to_contemporaneous(ts):
    """
    Simplify the ts to only the contemporaneous samples, and return the new ts + node map
    """
    samples = ts.samples()
    contmpr_samples = samples[ts.tables.nodes.time[samples] == 0]
    return ts.simplify(
        contmpr_samples, map_nodes=True, keep_unary=True, filter_populations=False,
        filter_sites=False, record_provenance=False, filter_individuals=False)


def get_unconstrained_times(ts, constrain_with_historic=True):
    """
    Returns the unconstrained times associated with each site.
    If multiple mutations are present at a site, use only the oldest mutation's node
    time. 
    If constrain_historic is True, ensure ancient individuals 
    """
    sites_time = np.zeros(ts.num_sites)
    for site in ts.sites():
        for mutation in site.mutations: 
            if sites_time[site.id] < node_ages[mutation.node]:
                sites_time[site.id] = node_ages[mutation.node]
    if np.any(np.bitwise_and(ts.tables.nodes.flags,
        tsinfer.formats.constants.NODE_IS_HISTORIC_SAMPLE)):
        individuals_metadata = tskit.unpack_bytes(output_ts.tables.individuals.metadata,
                output_ts.tables.individuals.metadata_offset)
        samples_times = np.zeroes(ts.num_samples)
        for historic_node in np.where(np.bitwise_and(ts.tables.nodes.flags, tsinfer.formats.constants.NODE_IS_HISTORIC_SAMPLE))[0]:
            samples_times[historic_node] = json.loads(individuals_metadata[historic_node].decode())["sample_data_time"]
        ancients = np.where(samples_times != 0)[0]
        ancient_samples_times = samples_times[ancients]
        for variant in ts.variants(samples=ancients):
            if np.any(variant.genotypes == 1):
                ancient_bound = np.max(ancient_samples_times[variant.genotypes == 1])
                if ancient_bound > sites_time[variant.site.id]:
                    sites_time[variant.site.id] = ancient_bound
    return sites_time

def preprocess_input(ts, minimum_gap=1000000):
    """
    Function to remove gaps from tree sequence for dating.
    Large regions without data will cause overflow/underflow errors in the
    inside-outside algorithm.
    """
    sites = ts.tables.sites.position[:]
    ts_trimmed = ts.keep_intervals([[sites[0], sites[-1] + 1]], simplify=False)
    assert np.array_equal(sites, ts_trimmed.tables.sites.position[:])
    gaps = np.argsort(sites[1:] - sites[:-1])
    delete_intervals = []
    for gap in gaps:
        gap_start = sites[gap] + 1
        gap_end = sites[gap + 1]
        # CONTINUE HERE
