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

import json
import numpy as np
import logging

import tskit

from . import base, provenance

logger = logging.getLogger(__name__)


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


def preprocess_ts(tree_sequence, minimum_gap=1000000, trim_telomeres=True):
    """
    Function to remove gaps without sites from tree sequence.
    Large regions without data can cause overflow/underflow errors in the
    inside-outside algorithm and poor performance more generally.
    Removed regions are recorded in the provenance of the resulting tree
    sequence.

    :param TreeSequence tree_sequence: The input :class`tskit.TreeSequence`
        to be trimmed.
    :param float minimum_gap: The minimum gap between sites to trim from the tree
        sequence. Default: "1000000"
    :param bool trim_telomeres: Should all material before the first site and after the
        last site be trimmed, regardless of the length. Default: "True"
    :return: A tree sequence with gaps removed.
    :rtype: tskit.TreeSequence
    """
    logger.info("Beginning preprocessing")
    logger.info("Minimum_gap: {} and trim_telomeres: {}".format(
        minimum_gap, trim_telomeres))
    if tree_sequence.num_sites < 1:
        raise ValueError(
                "Invalid tree sequence: no sites present")

    sites = tree_sequence.tables.sites.position[:]
    delete_intervals = []
    if trim_telomeres:
        first_site = sites[0] - 1
        if first_site > 0:
            delete_intervals.append([0, first_site])
            logger.info("TRIMMING TELOMERE: Snip topology "
                        "from 0 to first site at {}.".format(
                            first_site))
        last_site = sites[-1] + 1
        sequence_length = tree_sequence.get_sequence_length()
        if last_site < sequence_length:
            delete_intervals.append([last_site, sequence_length])
            logger.info("TRIMMING TELOMERE: Snip topology "
                        "from {} to end of sequence at {}.".format(
                            last_site, sequence_length))
    gaps = sites[1:] - sites[:-1]
    threshold_gaps = np.where(gaps >= minimum_gap)[0]
    for gap in threshold_gaps:
        gap_start = sites[gap] + 1
        gap_end = sites[gap + 1] - 1
        if gap_end > gap_start:
            logger.info("Gap Size is {}. Snip topology "
                        "from {} to {}.".format(
                            gap_end - gap_start, gap_start, gap_end))
            delete_intervals.append([gap_start, gap_end])
    delete_intervals = sorted(delete_intervals, key=lambda x: x[0])
    if len(delete_intervals) > 0:
        tree_sequence_trimmed = tree_sequence.delete_intervals(delete_intervals)
        tree_sequence_trimmed = tree_sequence_trimmed.simplify(filter_sites=False)
        assert tree_sequence.num_sites == tree_sequence_trimmed.num_sites
        return provenance.record_provenance(
                tree_sequence_trimmed, "preprocess_ts", minimum_gap=minimum_gap,
                trim_telomeres=trim_telomeres, delete_intervals=delete_intervals)
    else:
        logger.info("No gaps to trim")
        return tree_sequence


def get_site_times(tree_sequence, unconstrained=False, constrain_historic=True,
                   samples=None, mutation_age=False):
    """
    Returns the estimated time of the node of oldest mutation associated with each site.
    If multiple mutations are present at a site, use the oldest mutation's node
    time.
    If constrain_historic is True, check that the site age is at least as old as the
    oldest historic sample carrying a derived allele at that site.

    :param TreeSequence tree_sequence: The input :class`tskit.TreeSequence`.
    :param bool unconstrained: Use node ages which are unconstrained by site topology.
        Only applies when the inside-outside algorithm is used. Default: "True"
    :param bool constrain_historic: If true, ensure that the estimated age of sites
        where any historic sample
        Default: "True"
    :param tsinfer.formats.SampleData samples: A tsinfer SampleData object to
        add site times to. If both a SampleData file is provided and constrain_historic
        is True, use any historic individuals in this SampleData file to constrain site
        times.
    :param bool mutation_age: If true, return the age of the oldest mutation associated
        with each site.
        Default: "True"
    :return: An array of length tree_sequence.num_sites giving the estimated site time
        of each site.
    :rtype numpy.array

    """
    if tree_sequence.num_sites < 1:
        raise ValueError("Invalid tree sequence: no sites present")
    # Epsilon value to date sites with > 1 mutation where all mutations are singletons
    eps = 1e-6
    sites_time = np.empty(tree_sequence.num_sites)
    sites_time[:] = -np.inf
    node_ages = tree_sequence.tables.nodes.time[:]
    if unconstrained:
        metadata = tree_sequence.tables.nodes.metadata[:]
        metadata_offset = tree_sequence.tables.nodes.metadata_offset[:]
        for index, met in enumerate(tskit.unpack_bytes(metadata, metadata_offset)):
            if index not in tree_sequence.samples():
                try:
                    node_ages[index] = json.loads(met.decode())["mn"]
                except json.decoder.JSONDecodeError:
                    raise ValueError(
                            "Tree Sequence must be dated to use unconstrained=True")

    for tree in tree_sequence.trees():
        for site in tree.sites():
            for mutation in site.mutations:
                if mutation_age:
                    parent_node = tree.parent(mutation.node)
                    age = (node_ages[mutation.node] + node_ages[parent_node]) / 2
                else:
                    age = node_ages[mutation.node]
                if sites_time[site.id] < age:
                    sites_time[site.id] = age
            if len(site.mutations) > 1 and sites_time[site.id] == 0:
                sites_time[site.id] = eps
    if samples is not None and constrain_historic:
        assert samples.num_sites == tree_sequence.num_sites
        assert np.array_equal(samples.sites_position[:],
                              tree_sequence.tables.sites.position)
        samples_time = samples.individuals_time[:][samples.samples_individual[:]]
        ancient_samples = np.where(samples_time != 0)[0]
        ancient_samples_times = samples_time[samples_time != 0]
        for var in samples.variants():
            ancient_genos = var.genotypes[ancient_samples]
            if np.any(ancient_genos == 1):
                ancient_bound = np.max(ancient_samples_times[ancient_genos == 1])
                if ancient_bound > sites_time[var.site.id]:
                    sites_time[var.site.id] = ancient_bound
    elif np.any(np.bitwise_and(tree_sequence.tables.nodes.flags,
                               base.NODE_IS_HISTORIC_SAMPLE)) and constrain_historic:
        individuals_metadata = tskit.unpack_bytes(
                tree_sequence.tables.individuals.metadata,
                tree_sequence.tables.individuals.metadata_offset)
        samples_times = np.zeros(tree_sequence.num_samples)
        for historic_node in np.where(np.bitwise_and(
                tree_sequence.tables.nodes.flags, base.NODE_IS_HISTORIC_SAMPLE))[0]:
            samples_times[historic_node] = json.loads(
                    individuals_metadata[historic_node].decode())["sample_data_time"]
        ancients = np.where(samples_times != 0)[0]
        ancient_samples_times = samples_times[ancients]
        for variant in tree_sequence.variants(samples=ancients.astype('int32')):
            if np.any(variant.genotypes == 1):
                ancient_bound = np.max(ancient_samples_times[variant.genotypes == 1])
                if ancient_bound > sites_time[variant.site.id]:
                    sites_time[variant.site.id] = ancient_bound

    if samples is not None:
        copy = samples.copy()
        copy.sites_time[:] = sites_time
        copy.finalise()
        return copy
    else:
        return sites_time
