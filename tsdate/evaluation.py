# MIT License
#
# Copyright (c) 2021-23 Tskit Developers
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
Tools for comparing node times between tree sequences with different node sets
"""
import copy
import json
from collections import defaultdict
from itertools import product

import numpy as np
import scipy.sparse
import tskit
from math import isqrt

import matplotlib.pyplot as plt


class CladeMap:
    """
    An iterator across trees that maintains a mapping from a clade (a `frozenset` of
    sample IDs) to a `set` of nodes. When there are unary nodes, there may be multiple
    nodes associated with each clade.
    """

    def __init__(self, ts):
        self._nil = frozenset()
        self._nodes = defaultdict(set)  # nodes[clade] = {node ids}
        self._clades = defaultdict(frozenset)  # clades[node] = {sample ids}
        self.tree_sequence = ts
        self.tree = ts.first(sample_lists=True)
        for node in self.tree.nodes():
            clade = frozenset(self.tree.samples(node))
            self._nodes[clade].add(node)
            self._clades[node] = clade
        self._prev = copy.deepcopy(self._clades)
        self._diff = ts.edge_diffs()
        next(self._diff)

    def _propagate(self, edge, downdate=False):
        """
        Traverse path from `edge.parent` to root, either adding or removing the
        state (clade) associated with `edge.child` from the state of each
        visited node. Return a set with the node ids encountered during
        traversal.
        """
        nodes = set()
        node = edge.parent
        clade = self._clades[edge.child]
        while node != tskit.NULL:
            last = self._clades[node]
            self._clades[node] = last - clade if downdate else last | clade
            if len(last):
                self._nodes[last].remove(node)
                if len(self._nodes[last]) == 0:
                    del self._nodes[last]
            self._nodes[self._clades[node]].add(node)
            nodes.add(node)
            node = self.tree.parent(node)
        return nodes

    def next(self):  # noqa: A003
        """
        Advance to the next tree, returning the difference between trees as a
        dictionary of the form `node : (last_clade, next_clade)`
        """
        nodes = set()  # nodes with potentially altered clades
        diff = {}  # diff[node] = (prev_clade, curr_clade)

        if self.tree.index + 1 == self.tree_sequence.num_trees:
            return None

        # Subtract clades subtended by outgoing edges
        edge_diff = next(self._diff)
        for eo in edge_diff.edges_out:
            nodes |= self._propagate(eo, downdate=True)

        # Prune nodes that are no longer in tree
        for node in self._nodes[self._nil]:
            diff[node] = (self._prev[node], self._nil)
            del self._clades[node]
        nodes -= self._nodes[self._nil]
        self._nodes[self._nil].clear()

        # Add clades subtended by incoming edges
        self.tree.next()
        for ei in edge_diff.edges_in:
            nodes |= self._propagate(ei, downdate=False)

        # Find difference in clades between adjacent trees
        for node in nodes:
            diff[node] = (self._prev[node], self._clades[node])
            if self._prev[node] == self._clades[node]:
                del diff[node]

        # Sync previous and current states
        for node, (_, curr) in diff.items():
            if curr == self._nil:
                del self._prev[node]
            else:
                self._prev[node] = curr

        return diff

    @property
    def interval(self):
        """
        Return interval spanned by tree
        """
        return self.tree.interval

    def clades(self):
        """
        Return set of clades in tree
        """
        return self._nodes.keys() - self._nil

    def __getitem__(self, clade):
        """
        Return set of nodes associated with a given clade.
        """
        return frozenset(self._nodes[clade]) if frozenset(clade) in self else self._nil

    def __contains__(self, clade):
        """
        Check if a clade is present in the tree
        """
        return clade in self._nodes


def shared_node_spans(ts, other):
    """
    Calculate the spans over which pairs of nodes in two tree sequences are
    ancestral to indentical sets of samples.

    Returns a sparse matrix where rows correspond to nodes in `ts` and columns
    correspond to nodes in `other`.
    """

    if ts.sequence_length != other.sequence_length:
        raise ValueError("Tree sequences must be of equal sequence length.")

    if ts.num_samples != other.num_samples:
        raise ValueError("Tree sequences must have the same numbers of samples.")

    nil = frozenset()

    # Initialize clade iterators
    query = CladeMap(ts)
    target = CladeMap(other)

    # Initialize buffer[clade] = (query_nodes, target_nodes, left_coord)
    modified = query.clades() | target.clades()
    buffer = {}

    # Build sparse matrix of matches in triplet format
    query_node = []
    target_node = []
    shared_span = []
    right = 0
    while True:
        left = right
        right = min(query.interval[1], target.interval[1])

        # Flush pairs of nodes that no longer have matching clades
        for clade in modified:  # flush:
            if clade in buffer:
                n_i, n_j, start = buffer.pop(clade)
                span = left - start
                for i, j in product(n_i, n_j):
                    query_node.append(i)
                    target_node.append(j)
                    shared_span.append(span)

        # Add new pairs of nodes with matching clades
        for clade in modified:
            assert clade not in buffer
            if clade in query and clade in target:
                n_i, n_j = query[clade], target[clade]
                buffer[clade] = (n_i, n_j, left)

        if right == ts.sequence_length:
            break

        # Find difference in clades with advance to next tree
        modified.clear()
        for clade_map in (query, target):
            if clade_map.interval[1] == right:
                clade_diff = clade_map.next()
                for prev, curr in clade_diff.values():
                    if prev != nil:
                        modified.add(prev)
                    if curr != nil:
                        modified.add(curr)

    # Flush final tree
    for clade in buffer:
        n_i, n_j, start = buffer[clade]
        span = right - start
        for i, j in product(n_i, n_j):
            query_node.append(i)
            target_node.append(j)
            shared_span.append(span)

    numer = scipy.sparse.coo_matrix(
        (shared_span, (query_node, target_node)),
        shape=(ts.num_nodes, other.num_nodes),
    ).tocsr()

    return numer


def match_node_ages(ts, other):
    """
    For each node in `ts`, return the age of a matched node from `other`.  Node
    matching is accomplished by calculating the intervals over which pairs of
    nodes (one from `ts`, one from `other`) subtend the same set of samples.

    Returns three vectors of length `ts.num_nodes`: the age of the best
    matching node in `other` (e.g.  with the longest shared span); the
    proportion of the node span in `ts` that is covered by the best match; and
    the id of the best match in `other`.

    If either tree sequence contains unary nodes, then there may be multiple
    matches with the same span for a single node. In this case, the returned
    match is the node with the smallest integer id.
    """

    shared_spans = shared_node_spans(ts, other)
    matched_span = shared_spans.max(axis=1).todense().A1
    best_match = shared_spans.argmax(axis=1).A1
    # NB: if there are multiple nodes with the largest span in a row,
    # argmax returns the node with the smallest integer id
    matched_time = other.nodes_time[best_match]

    best_match[matched_span == 0] = tskit.NULL
    matched_time[matched_span == 0] = np.nan

    return matched_time, matched_span, best_match


def node_coverage(ts, inferred_ts, alpha):
    assert np.all(np.logical_and(1 > alpha, alpha > 0))
    posteriors = np.zeros((inferred_ts.num_nodes, 2))
    for n in inferred_ts.nodes():
        mn = json.loads(n.metadata or '{"mn":0}')["mn"]
        vr = json.loads(n.metadata or '{"vr":0}')["vr"]
        posteriors[n.id] = [mn**2 / vr, mn / vr] if vr > 0 else np.nan
    positions = {p: i for i, p in enumerate(ts.sites_position)}
    true_child = np.full(ts.sites_position.size, tskit.NULL)
    infr_child = np.full(ts.sites_position.size, tskit.NULL)
    for s in ts.sites():
        if len(s.mutations) == 1:
            sid = positions[s.position]
            true_child[sid] = s.mutations[0].node
    for s in inferred_ts.sites():
        if len(s.mutations) == 1:
            sid = positions[s.position]
            nid = s.mutations[0].node
            if not np.isnan(posteriors[nid, 0]):
                infr_child[s.id] = s.mutations[0].node
    missing = np.logical_or(true_child == tskit.NULL, infr_child == tskit.NULL)
    infr_child = infr_child[~missing]
    true_child = true_child[~missing]
    post = posteriors[infr_child]
    upper = np.zeros((post.shape[0], alpha.size))
    lower = np.zeros((post.shape[0], alpha.size))
    for i in range(post.shape[0]):
        shape, rate = post[i, 0], post[i, 1]
        if shape <= 1:
            upper[i] = scipy.stats.gamma.ppf(1 - alpha, shape, scale=1 / rate)
            lower[i] = 0.0
        else:
            upper[i] = scipy.stats.gamma.ppf(1 - alpha / 2, shape, scale=1 / rate)
            lower[i] = scipy.stats.gamma.ppf(alpha / 2, shape, scale=1 / rate)
    true = ts.nodes_time[true_child]
    is_covered = np.logical_and(
        true[:, np.newaxis] < upper, true[:, np.newaxis] > lower
    )
    prop_covered = np.sum(is_covered, axis=0) / is_covered.shape[0]
    # import matplotlib.pyplot as plt
    # plt.axline((0,0), slope=1, linestyle="--", color="black")
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.xlabel("Expected coverage")
    # plt.xlabel("Observed coverage")
    # plt.scatter(1 - alpha, prop_covered, color="red")
    # plt.savefig(plot)
    # plt.clf()
    # plt.clf()
    # fig, axs = plt.subplots(1, figsize=(10,5))
    # cmap = plt.get_cmap("plasma")
    # samp = np.random.randint(0, true.size, size=1000)
    # rnk = scipy.stats.rankdata(true[samp])
    # for i in range(alpha.size):
    #    axs.vlines(
    #        x=rnk,
    #        ymin=np.log10(lower[samp, i]) - np.log10(true[samp]),
    #        ymax=np.log10(upper[samp, i]) - np.log10(true[samp]),
    #        color=cmap(i/(alpha.size-1)),
    #        linewidth=1,
    #    )
    # axs.axhline(y=0, linestyle="--", color="black")
    # axs.set_xlabel("True age rank order")
    # axs.set_ylabel("Interval - true age (log)")
    # plt.savefig("bar.png")
    # plt.clf()
    return prop_covered


def mutation_coverage(ts, inferred_ts, alpha):
    assert np.all(np.logical_and(1 > alpha, alpha > 0))
    # extract mutation posteriors from metadata
    posteriors = np.zeros((inferred_ts.num_mutations, 2))
    for m in inferred_ts.mutations():
        mn = json.loads(m.metadata or '{"mn":0}')["mn"]
        vr = json.loads(m.metadata or '{"vr":0}')["vr"]
        posteriors[m.id] = [mn**2 / vr, mn / vr] if vr > 0 else np.nan
    # find shared biallelic sites
    positions = {p: i for i, p in enumerate(ts.sites_position)}
    true_mut = np.full(ts.sites_position.size, tskit.NULL)
    infr_mut = np.full(ts.sites_position.size, tskit.NULL)
    for s in ts.sites():
        if len(s.mutations) == 1:
            sid = positions[s.position]
            true_mut[sid] = s.mutations[0].id
    for s in inferred_ts.sites():
        if len(s.mutations) == 1:
            mid = s.mutations[0].id
            if not np.isnan(posteriors[mid, 0]):
                sid = positions[s.position]
                infr_mut[sid] = s.mutations[0].id
    missing = np.logical_or(true_mut == tskit.NULL, infr_mut == tskit.NULL)
    infr_mut = infr_mut[~missing]
    true_mut = true_mut[~missing]
    # calculate coverage
    post = posteriors[infr_mut]
    upper = np.zeros((post.shape[0], alpha.size))
    lower = np.zeros((post.shape[0], alpha.size))
    for i in range(post.shape[0]):
        shape, rate = post[i, 0], post[i, 1]
        if shape <= 1:
            upper[i] = scipy.stats.gamma.ppf(1 - alpha, shape, scale=1 / rate)
            lower[i] = 0.0
        else:
            upper[i] = scipy.stats.gamma.ppf(1 - alpha / 2, shape, scale=1 / rate)
            lower[i] = scipy.stats.gamma.ppf(alpha / 2, shape, scale=1 / rate)
    true = ts.mutations_time[true_mut]
    is_covered = np.logical_and(
        true[:, np.newaxis] < upper, true[:, np.newaxis] > lower
    )
    prop_covered = np.sum(is_covered, axis=0) / is_covered.shape[0]
    # plt.clf()
    # plt.axline((0,0), slope=1, linestyle="--", color="black")
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.xlabel("Expected coverage")
    # plt.xlabel("Observed coverage")
    # plt.scatter(1 - alpha, prop_covered, color="red")
    # plt.savefig(plot)
    # plt.clf()
    # fig, axs = plt.subplots(1, figsize=(10,5))
    # cmap = plt.get_cmap("plasma")
    # samp = np.random.randint(0, true.size, size=1000)
    # rnk = scipy.stats.rankdata(true[samp])
    # for i in range(alpha.size):
    #    axs.vlines(
    #        x=rnk,
    #        ymin=np.log10(lower[samp, i]) - np.log10(true[samp]),
    #        ymax=np.log10(upper[samp, i]) - np.log10(true[samp]),
    #        color=cmap(i/(alpha.size-1)),
    #        linewidth=1,
    #    )
    # axs.axhline(y=0, linestyle="--", color="black")
    # axs.set_xlabel("True age rank order")
    # axs.set_ylabel("Interval - true age (log)")
    # plt.savefig("foo.png")
    # plt.clf()
    return prop_covered


def allele_frequency_spectra(ts, mutation_rate, plotpath=None, title=None, max_freq=None, num_bins=9, num_windows=500, polarised=True):
    """
    Calculate site and branch allele frequency spectra across windows, where
    adjacent AFS bins are pooled. Optionally produce a scatterplot for each
    pooled bin. Optionally truncate the AFS at a given `max_freq`.
    """

    if max_freq is None:
        max_freq = -1
    ts_trim = ts.trim()
    windows = np.linspace(0, ts_trim.sequence_length, num_windows + 1)
    site_afs = ts_trim.allele_frequency_spectrum(
        mode='site', windows=windows, span_normalise=False, polarised=polarised
    )[:, 1:max_freq]
    branch_afs = mutation_rate * ts_trim.allele_frequency_spectrum(
        mode='branch', windows=windows, span_normalise=False, polarised=polarised
    )[:, 1:max_freq]
    dim = isqrt(num_bins)
    num_bins = dim * dim
    cumulative = np.arange(0, branch_afs.shape[1], dtype=np.float64)
    cumulative /= cumulative[-1]
    bins = np.linspace(0, 1, num_bins + 1)
    bins = np.searchsorted(cumulative, bins, side='right') - 1
    if plotpath is not None:
        fig, axs = plt.subplots(dim, dim, squeeze=0)
        fudge = 90 / 100
        for i, j, ax in zip(bins[:-1], bins[1:], axs.reshape(-1)):
            obs = site_afs[:, i:j].sum(axis=1)
            exp = branch_afs[:, i:j].sum(axis=1)
            ax.text(0.02, 0.98, f"{i+1}:{j+1}", ha='left', va='top', transform=ax.transAxes, size=8)
            ax.set_xticks(np.linspace(exp.min(), exp.max(), 3))
            ax.set_yticks(np.linspace(obs.min(), obs.max(), 3))
            ax.set_xlim(exp.min() * fudge, exp.max() / fudge)
            ax.set_ylim(obs.min() * fudge, obs.max() / fudge)
            ax.tick_params(labelsize=8)
            ax.scatter(exp, obs, color="firebrick", s=4)
            ax.axline((np.mean(obs), np.mean(obs)), slope=1, linestyle="--", color="black")
        fig.supylabel("Observed # sites in window")
        fig.supxlabel("Expected # sites in window")
        if title is not None:
            fig.suptitle(title)
        plt.tight_layout()
        plt.savefig(plotpath)
        plt.clf()
    return site_afs, branch_afs
