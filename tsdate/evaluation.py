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
from itertools import groupby
from itertools import product
from math import isqrt

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import tskit


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


# --- infrastructure for testing against polytomies --- #


def remove_edges(ts, edge_id_remove_list):
    edges_to_remove_by_child = defaultdict(list)
    edge_id_remove_list = set(edge_id_remove_list)
    for m in ts.mutations():
        if m.edge in edge_id_remove_list:
            # If we remove this edge, we will remove the associated mutation
            # as the child node won't have ancestral material in this region.
            # So we force the user to explicitly (re)move the mutations beforehand
            raise ValueError("Cannot remove edges that have associated mutations")
    for remove_edge in edge_id_remove_list:
        e = ts.edge(remove_edge)
        edges_to_remove_by_child[e.child].append(e)

    # sort left-to-right for each child
    for k, v in edges_to_remove_by_child.items():
        edges_to_remove_by_child[k] = sorted(v, key=lambda e: e.left)
        # check no overlaps
        for e1, e2 in zip(edges_to_remove_by_child[k], edges_to_remove_by_child[k][1:]):
            assert e1.right <= e2.left

    # Sanity check: this means the topmost node will deal with modified edges
    # left at the end
    assert ts.edge(-1).parent not in edges_to_remove_by_child

    new_edges = defaultdict(list)
    tables = ts.dump_tables()
    tables.edges.clear()
    # Edges are sorted by parent time, youngest first, so we can iterate over
    # nodes-as-parents visiting children before parents by using itertools.groupby
    for parent_id, ts_edges in groupby(ts.edges(), lambda e: e.parent):
        # Iterate through the ts edges *plus* the polytomy edges we created in
        # previous steps.  This allows us to re-edit polytomy edges when the
        # edges_to_remove are stacked
        edges = list(ts_edges)
        if parent_id in new_edges:
            edges += new_edges.pop(parent_id)
        if parent_id in edges_to_remove_by_child:
            for e in edges:
                assert parent_id == e.parent
                left = -1
                if e.id in edge_id_remove_list:
                    continue
                # NB: we go left to right along the target edges, reducing edge
                # e as required
                for target_edge in edges_to_remove_by_child[parent_id]:
                    # As we go along the target_edges, gradually split e into
                    # chunks.  If edge e is in the target_edge region, change
                    # the edge parent
                    assert target_edge.left > left
                    left = target_edge.left
                    if e.left >= target_edge.right:
                        # This target edge is entirely to the LHS of edge e,
                        # with no overlap
                        continue
                    elif e.right <= target_edge.left:
                        # This target edge is entirely to the RHS of edge e
                        # with no overlap.  Since target edges are sorted by
                        # left coord, all other target edges are to RHS too,
                        # and we are finished dealing with edge e
                        tables.edges.append(e)
                        e = None
                        break
                    else:
                        # Edge e must overlap with current target edge somehow
                        if e.left < target_edge.left:
                            # Edge had region to LHS of target
                            # Add the left hand section (change the edge right coord)
                            tables.edges.add_row(
                                left=e.left,
                                right=target_edge.left,
                                parent=e.parent,
                                child=e.child,
                            )
                            e = e.replace(left=target_edge.left)
                        if e.right > target_edge.right:
                            # Edge continues after RHS of target
                            assert e.left < target_edge.right
                            new_edges[target_edge.parent].append(
                                e.replace(
                                    right=target_edge.right, parent=target_edge.parent
                                )
                            )
                            e = e.replace(left=target_edge.right)
                        else:
                            # No more of edge e to RHS
                            assert e.left < e.right
                            new_edges[target_edge.parent].append(
                                e.replace(parent=target_edge.parent)
                            )
                            e = None
                            break
                if e is not None:
                    # Need to add any remaining regions of edge back in
                    tables.edges.append(e)
        else:
            # NB: sanity check at top means that the oldest node will have no
            # edges above, so the last iteration should hit this branch
            for e in edges:
                if e.id not in edge_id_remove_list:
                    tables.edges.append(e)
    assert len(new_edges) == 0
    tables.sort()
    return tables.tree_sequence()


def unsupported_edges(ts, per_interval=False):
    """
    Return the internal edges that are unsupported by a mutation.
    If ``per_interval`` is True, each interval needs to be supported,
    otherwise, a mutation on an edge (even if there are multiple intervals
    per edge) will result in all intervals on that edge being treated
    as supported.
    """
    edges_to_remove = np.ones(ts.num_edges, dtype="bool")
    edges_to_remove[[m.edge for m in ts.mutations()]] = False
    # We don't remove edges above samples
    edges_to_remove[np.isin(ts.edges_child, ts.samples())] = False

    if per_interval:
        return np.where(edges_to_remove)[0]
    else:
        keep = ~edges_to_remove
        for p, c in zip(ts.edges_parent[keep], ts.edges_child[keep]):
            edges_to_remove[
                np.logical_and(ts.edges_parent == p, ts.edges_child == c)
            ] = False
        return np.where(edges_to_remove)[0]


# --- first drafts of diagnostic plots --- #


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


def mutations_time(ts, inferred_ts, min_freq=None, max_freq=None, plotpath=None):
    """
    Return true and inferred mutation ages, optionally creating a scatterplot and
    filtering by minimum or maximum frequency.
    """
    # find shared biallelic sites
    positions = {p: i for i, p in enumerate(ts.sites_position)}
    true_mut = np.full(ts.sites_position.size, tskit.NULL)
    infr_mut = np.full(ts.sites_position.size, tskit.NULL)
    for s in ts.sites():
        if len(s.mutations) == 1:
            if s.mutations[0].edge != tskit.NULL:
                sid = positions[s.position]
                true_mut[sid] = s.mutations[0].id
    for s in inferred_ts.sites():
        if len(s.mutations) == 1:
            if s.mutations[0].edge != tskit.NULL:
                sid = positions[s.position]
                infr_mut[sid] = s.mutations[0].id
    missing = np.logical_or(true_mut == tskit.NULL, infr_mut == tskit.NULL)
    infr_mut = infr_mut[~missing]
    true_mut = true_mut[~missing]
    mean = inferred_ts.mutations_time[infr_mut]
    truth = ts.mutations_time[true_mut]
    # filter by frequency
    if min_freq is not None or max_freq is not None:
        freq = np.zeros(inferred_ts.num_mutations)
        for t in inferred_ts.trees():
            for m in t.mutations():
                freq[m.id] = t.num_samples(m.node)
        if min_freq is None:
            min_freq = np.min(freq)
        if max_freq is None:
            max_freq = np.max(freq)
        freq = freq[infr_mut]
        is_freq = np.logical_and(freq >= min_freq, freq <= max_freq)
        mean = mean[is_freq]
        truth = truth[is_freq]
    # plot
    if plotpath is not None:
        rsq = np.corrcoef(np.log10(mean), np.log10(truth))[0, 1] ** 2
        bias = np.mean(np.log10(mean) - np.log10(truth))
        pt1 = (truth.mean(), truth.mean())
        pt2 = (truth.mean() + 1, truth.mean() + 1)
        info = f"$r^2 = {rsq:0.3f}$\n$\\mathrm{{bias}} = {bias:0.3f}$"
        plt.hexbin(truth, mean, xscale="log", yscale="log", mincnt=1)
        plt.text(0.01, 0.99, info, ha="left", va="top", transform=plt.gca().transAxes)
        plt.axline(pt1, pt2, linestyle="--", color="firebrick")
        plt.xlabel("True mutation age")
        plt.ylabel("Estimated mutation age")
        plt.tight_layout()
        plt.savefig(plotpath)
        plt.clf()
    return truth, mean


def afs_bias(ts, mutation_rate, plotpath=None, polarised=True):
    """
    Calculate site and branch allele frequency spectra across windows, where
    adjacent AFS bins are pooled. Optionally produce a scatterplot for each
    pooled bin. Optionally truncate the AFS at a given `max_freq`.
    """
    ts_trim = ts.trim()
    site_afs = ts_trim.allele_frequency_spectrum(
        mode="site", span_normalise=False, polarised=polarised
    )
    branch_afs = mutation_rate * ts_trim.allele_frequency_spectrum(
        mode="branch", span_normalise=False, polarised=polarised
    )
    if plotpath is not None:
        plt.scatter(np.arange(site_afs.size), site_afs, c="black", s=8)
        plt.scatter(np.arange(site_afs.size), branch_afs, c="firebrick", s=8)
        plt.xlabel("Mutation frequency")
        plt.ylabel("# mutations")
        plt.yscale("log")
        plt.savefig(plotpath)
        plt.clf()
    return site_afs, branch_afs


def allele_frequency_spectra(
    ts,
    mutation_rate,
    plotpath=None,
    title=None,
    max_freq=None,
    num_bins=9,
    num_windows=500,
    polarised=True,
    size_biased=False,
):
    """
    Calculate site and branch allele frequency spectra across windows, where
    adjacent AFS bins are pooled. Optionally produce a scatterplot for each
    pooled bin. Optionally truncate the AFS at a given `max_freq`.
    """

    if max_freq is None:
        max_freq = -1
    ts_trim = ts.trim()
    windows = np.linspace(0, ts_trim.sequence_length, num_windows + 1)
    if size_biased:
        bin_sizes = np.tile(np.arange(ts.num_samples + 1), (num_windows, 1))
    else:
        bin_sizes = np.ones((num_windows, ts.num_samples + 1))
    site_afs = bin_sizes * ts_trim.allele_frequency_spectrum(
        mode="site", windows=windows, span_normalise=False, polarised=polarised
    )
    site_afs = site_afs[:, 1:max_freq]
    branch_afs = (
        mutation_rate
        * bin_sizes
        * ts_trim.allele_frequency_spectrum(
            mode="branch", windows=windows, span_normalise=False, polarised=polarised
        )
    )
    branch_afs = branch_afs[:, 1:max_freq]
    dim = isqrt(num_bins)
    num_bins = dim * dim
    cumulative = np.arange(0, branch_afs.shape[1], dtype=np.float64)
    cumulative /= cumulative[-1]
    bins = np.linspace(0, 1, num_bins + 1)
    bins = np.searchsorted(cumulative, bins, side="right") - 1
    if plotpath is not None:
        fig, axs = plt.subplots(dim, dim, squeeze=0)
        fudge = 90 / 100
        for i, j, ax in zip(bins[:-1], bins[1:], axs.reshape(-1)):
            obs = site_afs[:, i:j].sum(axis=1)
            exp = branch_afs[:, i:j].sum(axis=1)
            ax.text(
                0.02,
                0.98,
                f"{i+1}:{j+1}",
                ha="left",
                va="top",
                transform=ax.transAxes,
                size=8,
            )
            ax.set_xticks(np.linspace(exp.min(), exp.max(), 3))
            ax.set_yticks(np.linspace(obs.min(), obs.max(), 3))
            ax.set_xlim(exp.min() * fudge, exp.max() / fudge)
            ax.set_ylim(obs.min() * fudge, obs.max() / fudge)
            ax.tick_params(labelsize=8)
            ax.scatter(exp, obs, color="firebrick", s=4)
            ax.axline(
                (np.mean(obs), np.mean(obs)), slope=1, linestyle="--", color="black"
            )
        fig.supylabel("Observed # sites in window")
        fig.supxlabel("Expected # sites in window")
        if title is not None:
            fig.suptitle(title)
        plt.tight_layout()
        plt.savefig(plotpath)
        plt.clf()
    return site_afs, branch_afs
