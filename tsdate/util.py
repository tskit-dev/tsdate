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
import logging
import warnings

import numpy as np
import tskit

from . import provenance

logger = logging.getLogger(__name__)


def get_single_root(tree):
    # TODO - use new 'root_threshold=2' to avoid having to check isolated nodes
    topological_roots = [r for r in tree.roots if tree.num_children(r) != 0]
    if len(topological_roots) > 1:
        raise ValueError(f"Invalid tree sequence: tree {tree.index} has >1 root")
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
        contmpr_samples,
        map_nodes=True,
        keep_unary=True,
        filter_populations=False,
        filter_sites=False,
        record_provenance=False,
        filter_individuals=False,
    )


def preprocess_ts(
    tree_sequence, *, minimum_gap=1000000, remove_telomeres=True, **kwargs
):
    """
    Function to prepare tree sequences for dating by removing gaps without sites and
    simplifying the tree sequence. Large regions without data can cause
    overflow/underflow errors in the inside-outside algorithm and poor performance more
    generally. Removed regions are recorded in the provenance of the resulting tree
    sequence.

    :param TreeSequence tree_sequence: The input :class`tskit.TreeSequence`
        to be preprocessed.
    :param float minimum_gap: The minimum gap between sites to remove from the tree
        sequence. Default: "1000000"
    :param bool remove_telomeres: Should all material before the first site and after the
        last site be removed, regardless of the length. Default: "True"
    :param \\**kwargs: All further keyword arguments are passed to the ``tskit.simplify``
        command.

    :return: A tree sequence with gaps removed.
    :rtype: tskit.TreeSequence
    """
    logger.info("Beginning preprocessing")
    logger.info(f"Minimum_gap: {minimum_gap} and remove_telomeres: {remove_telomeres}")
    if tree_sequence.num_sites < 1:
        raise ValueError("Invalid tree sequence: no sites present")

    sites = tree_sequence.tables.sites.position[:]
    delete_intervals = []
    if remove_telomeres:
        first_site = sites[0] - 1
        if first_site > 0:
            delete_intervals.append([0, first_site])
            logger.info(
                "REMOVING TELOMERE: Snip topology "
                "from 0 to first site at {}.".format(first_site)
            )
        last_site = sites[-1] + 1
        sequence_length = tree_sequence.get_sequence_length()
        if last_site < sequence_length:
            delete_intervals.append([last_site, sequence_length])
            logger.info(
                "REMOVING TELOMERE: Snip topology "
                "from {} to end of sequence at {}.".format(last_site, sequence_length)
            )
    gaps = sites[1:] - sites[:-1]
    threshold_gaps = np.where(gaps >= minimum_gap)[0]
    for gap in threshold_gaps:
        gap_start = sites[gap] + 1
        gap_end = sites[gap + 1] - 1
        if gap_end > gap_start:
            logger.info(
                "Gap Size is {}. Snip topology "
                "from {} to {}.".format(gap_end - gap_start, gap_start, gap_end)
            )
            delete_intervals.append([gap_start, gap_end])
    delete_intervals = sorted(delete_intervals, key=lambda x: x[0])
    if len(delete_intervals) > 0:
        tree_sequence_trimmed = tree_sequence.delete_intervals(
            delete_intervals, simplify=False
        )
        tree_sequence_trimmed = tree_sequence_trimmed.simplify(**kwargs)
        tree_sequence_trimmed = provenance.record_provenance(
            tree_sequence_trimmed,
            "preprocess_ts",
            minimum_gap=minimum_gap,
            remove_telomeres=remove_telomeres,
            delete_intervals=delete_intervals,
        )
    else:
        logger.info("No gaps to remove")
        tree_sequence_trimmed = tree_sequence.simplify(**kwargs)
    if tree_sequence.num_sites != tree_sequence_trimmed.num_sites:
        warnings.warn(
            "Different number of sites after preprocessing. "
            "Try using **{'filter_sites:' False} to avoid this",
            RuntimeWarning,
        )

    return tree_sequence_trimmed


def nodes_time(tree_sequence, unconstrained=True):
    nodes_age = tree_sequence.tables.nodes.time[:]
    if unconstrained:
        metadata = tree_sequence.tables.nodes.metadata[:]
        metadata_offset = tree_sequence.tables.nodes.metadata_offset[:]
        for index, met in enumerate(tskit.unpack_bytes(metadata, metadata_offset)):
            if index not in tree_sequence.samples():
                try:
                    nodes_age[index] = json.loads(met.decode())["mn"]
                except (KeyError, json.decoder.JSONDecodeError):
                    raise ValueError(
                        "Tree Sequence must be tsdated with the "
                        "Inside-Outside Method. Use unconstrained=False "
                        "if not."
                    )
    return nodes_age


def sites_time_from_ts(
    tree_sequence,
    *,
    unconstrained=True,
    mutation_age="child",
    ignore_multiallelic=True,
    eps=1e-6,
):
    """
    Returns the estimated time of the oldest mutation at each site.

    :param TreeSequence tree_sequence: The input :class`tskit.TreeSequence`.
    :param bool unconstrained: Use node ages which are unconstrained by site topology.
        Only applies when the inside-outside algorithm is used.
        Default: "True"
    :param str mutation_age: Defines how mutation times are calculated. Options are
        "child", "parent", "arithmetic" or "geometric". If "child", mutation ages are
        defined by the time of the mutation's child node.
        If "parent", mutation ages are the time of the mutation's parent node.
        If "arithmetic", mutation times are calculated using the arithmetic mean of
        the parent and child nodes of the mutation. If "geometric", use the geometric
        mean of the parent and child nodes of the mutation.
        Default: "child"
    :param bool ignore_multiallelic: If True, return the age of oldest mutation
        asscoated with any allele at multiallelic sites. If False, return
        tskit.UNKNOWN_TIME at multiallelic sites.
        Default: True
    :param float eps: Time value assigned to sites with > 1 mutation where the age of
        oldest mutation is 0. For example, this value would be assigned to sites where
        all mutations are singletons and the mutation_age parameter is assigned to
        "child" or "geometric".
        Default: 1e-6

    :return: Array of length tree_sequence.num_sites with estimated time of each site
    :rtype: numpy.array
    """
    if tree_sequence.num_sites < 1:
        raise ValueError("Invalid tree sequence: no sites present")
    if mutation_age not in ["arithmetic", "geometric", "child", "parent"]:
        raise ValueError(
            "mutation_age parameter must be 'arithmetic', 'geometric', 'child', or\
            'parent'"
        )
    sites_time = np.empty(tree_sequence.num_sites)
    sites_time[:] = -np.inf
    nodes_age = nodes_time(tree_sequence, unconstrained=unconstrained)

    for tree in tree_sequence.trees():
        for site in tree.sites():
            alleles = {site.ancestral_state}
            for mutation in site.mutations:
                alleles.add(mutation.derived_state)
                if mutation_age == "child":
                    age = nodes_age[mutation.node]
                else:
                    parent_age = nodes_age[tree.parent(mutation.node)]
                    if mutation_age == "parent":
                        age = parent_age
                    elif mutation_age == "arithmetic":
                        age = (nodes_age[mutation.node] + parent_age) / 2
                    elif mutation_age == "geometric":
                        age = np.sqrt(nodes_age[mutation.node] * parent_age)
                if sites_time[site.id] < age:
                    sites_time[site.id] = age
            if len(site.mutations) > 1 and sites_time[site.id] == 0:
                sites_time[site.id] = eps
            if ignore_multiallelic is False:
                if len(alleles) > 2:
                    sites_time[site.id] = tskit.UNKNOWN_TIME
    return sites_time


def add_sampledata_times(samples, sites_time):
    """
    Return a tsinfer.SampleData file with estimated times associated with sites.
    Ensures that each site's time is at least as old as the oldest historic sample
    carrying a derived allele at that site.

    :param tsinfer.formats.SampleData samples: A tsinfer SampleData object to
        add site times to. Any historic individuals in this SampleData file are used to
        constrain site times.

    :return: A tsinfer.SampleData file
    :rtype: tsinfer.SampleData
    """
    if samples.num_sites != len(sites_time):
        raise ValueError(
            "sites_time should contain the same number of sites as the SampleData file"
        )
    # Get constraints from ancients
    sites_bound = samples.min_site_times(individuals_only=True)
    # Use maximum of constraints and estimated site times
    sites_time = np.maximum(sites_time, sites_bound)
    copy = samples.copy()
    copy.sites_time[:] = sites_time
    copy.finalise()
    return copy
