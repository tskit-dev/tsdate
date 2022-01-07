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

    tables = tree_sequence.dump_tables()
    sites = tables.sites.position[:]
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
        sequence_length = tables.sequence_length
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
        tables.delete_intervals(delete_intervals, simplify=False)
        tables.simplify(**kwargs)
        provenance.record_provenance(
            tables,
            "preprocess_ts",
            minimum_gap=minimum_gap,
            remove_telomeres=remove_telomeres,
            delete_intervals=delete_intervals,
        )
    else:
        logger.info("No gaps to remove")
        tables.simplify(**kwargs)
    if tree_sequence.num_sites != tables.sites.num_rows:
        warnings.warn(
            "Different number of sites after preprocessing. "
            "Try using **{'filter_sites:' False} to avoid this",
            RuntimeWarning,
        )

    return tables.tree_sequence()


def nodes_time_unconstrained(tree_sequence):
    """
    Return the unconstrained node times for every node in a tree sequence that has
    been dated using ``tsdate`` with the inside-outside algorithm (these times are
    stored in the node metadata). Will produce an error if the tree sequence does
    not contain this information.
    """
    nodes_time = tree_sequence.tables.nodes.time.copy()
    metadata = tree_sequence.tables.nodes.metadata
    metadata_offset = tree_sequence.tables.nodes.metadata_offset
    for index, met in enumerate(tskit.unpack_bytes(metadata, metadata_offset)):
        if index not in tree_sequence.samples():
            try:
                nodes_time[index] = json.loads(met.decode())["mn"]
            except (KeyError, json.decoder.JSONDecodeError):
                raise ValueError(
                    "Tree Sequence must be tsdated with the Inside-Outside Method."
                )
    return nodes_time


def sites_time_from_ts(
    tree_sequence, *, unconstrained=True, node_selection="child", min_time=1
):
    """
    Returns an estimated "time" for each site. This is the estimated age of the oldest
    MRCA which possesses a derived variant at that site, and is useful for performing
    (re)inference of a tree sequence. It is calculated from the ages of nodes, with the
    appropriate nodes identified by the position of mutations in the trees.

    If node times in the tree sequence have been estimated by ``tsdate`` using the
    inside-outside algorithm, then as well as a time in the tree sequence, nodes will
    store additional time estimates that have not been explictly constrained by the
    tree topology. By default, this function tries to use these "unconstrained" times,
    although this is likely to fail (with a warning) on tree sequences that have not
    been processed by ``tsdate``: in this case the standard node times can be used by
    setting ``unconstrained=False``.

    The concept of a site time is meaningless for non-variable sites, and
    so the returned time for these sites is ``np.nan`` (note that this is not exactly
    the same as tskit.UNKNOWN_TIME, which marks sites that could have a meaningful time
    but whose time estimate is unknown).

    :param TreeSequence tree_sequence: The input :class`tskit.TreeSequence`.
    :param bool unconstrained: Use estimated node times which have not been constrained
        by tree topology. If ``True`` (default), this requires a tree sequence which has
        been dated using the ``tsdate`` inside-outside algorithm. If this is not the
        case, specify ``False`` to use the standard tree sequence node times.
    :param str node_selection: Defines how site times are calculated from the age of
        the upper and lower nodes that bound each mutation at the site. Options are
        "child", "parent", "arithmetic" or "geometric", with the following meanings

        * ``'child'`` (default): the site time is the age of the oldest node
          *below* each mutation at the site
        * ``'parent'``: the site time is the age of the oldest node *above* each
          mutation at the site
        * ``'arithmetic'``: the arithmetic mean of the ages of the node above and the
          node below each mutation is calculated; the site time is the oldest
          of these means.
        * ``'geometric'``: the geometric mean of the ages of the node above and the
          node below each mutation is calculated; the site time is the oldest
          of these means

    :param float min_time: A site time of zero implies that no MRCA in the past
        possessed the derived variant, so the variant cannot be used for inferring
        relationships between the samples. To allow all variants to be potentially
        available for inference, if a site time would otherwise be calculated as zero
        (for example, where the ``mutation_age`` parameter is "child" or "geometric"
        and all mutations at a site are associated with leaf nodes), a minimum site
        greater than 0 is recommended. By default this is set to 1, which is generally
        reasonable for times measured in generations or years, although it is also
        fine to set this to a small epsilon value.

    :return: Array of length tree_sequence.num_sites with estimated time of each site
    :rtype: numpy.array
    """
    if tree_sequence.num_sites < 1:
        raise ValueError("Invalid tree sequence: no sites present")
    if node_selection not in ["arithmetic", "geometric", "child", "parent"]:
        raise ValueError(
            "The node_selection parameter must be "
            "'child', 'parent', 'arithmetic', or 'geometric'"
        )
    if unconstrained:
        try:
            nodes_time = nodes_time_unconstrained(tree_sequence)
        except ValueError as e:
            e.args += "Try calling sites_time_from_ts() with unconstrained=False."
            raise
    else:
        nodes_time = tree_sequence.tables.nodes.time
    sites_time = np.full(tree_sequence.num_sites, np.nan)

    for tree in tree_sequence.trees():
        for site in tree.sites():
            for mutation in site.mutations:
                parent_node = tree.parent(mutation.node)
                if node_selection == "child" or parent_node == tskit.NULL:
                    age = nodes_time[mutation.node]
                else:
                    parent_age = nodes_time[parent_node]
                    if node_selection == "parent":
                        age = parent_age
                    elif node_selection == "arithmetic":
                        age = (nodes_time[mutation.node] + parent_age) / 2
                    elif node_selection == "geometric":
                        age = np.sqrt(nodes_time[mutation.node] * parent_age)
                if np.isnan(sites_time[site.id]) or sites_time[site.id] < age:
                    sites_time[site.id] = age
            if sites_time[site.id] < min_time:
                sites_time[site.id] = min_time
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
