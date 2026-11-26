"""
Expectation propagation state for the equivalence tests of ``tsdate.variational``:
snapshots of real state from ``ExpectationPropagation`` (editable, so that tests can
force branches), one-edge snapshots, and runners that call the ``propagate_*``
functions of a namespace on copies of a snapshot.
"""

import numpy as np
import tskit

from tsdate.variational import (
    CONSTRNT,
    LEAFWARD,
    LOWER,
    MIXPRIOR,
    ROOTWARD,
    UPPER,
    USE_BLOCK_LIKELIHOOD,
    USE_EDGE_LIKELIHOOD,
    ExpectationPropagation,
)

# fixture name -> (singletons_phased, mutation_rate)
SNAPSHOT_SPECS = {
    "small_ts": (True, 1e-8),
    "small_ts_unphased": (False, 1e-7),
    "small_ts_historical": (True, 1e-8),
}

_COPIED = (
    "edge_order",
    "edge_parents",
    "edge_children",
    "edge_likelihoods",
    "block_order",
    "block_likelihoods",
    "block_nodes",
    "node_constraints",
    "node_posterior",
    "edge_logconst",
    "block_logconst",
    "unconstrained_roots",
    "roots",
    "leaves",
    "mutation_order",
    "mutation_edges",
    "mutation_blocks",
)


class Snapshot:
    """
    Copies of the arrays of an ``ExpectationPropagation`` after some iterations, so
    that tests can edit them freely and rebuild fresh ``EPFactors`` of any class.
    """

    def __init__(self, ts, ep):
        self.ts = ts
        for name in _COPIED:
            setattr(self, name, getattr(ep, name).copy())
        factors = ep.factors
        self.fields = {k: getattr(factors, k).copy() for k in self.FIELDS}
        self.num_mutations = ep.mutation_phase.size

    FIELDS = ("node", "edge", "block", "scale")

    def copy(self):
        new = object.__new__(Snapshot)
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                v = v.copy()
            elif isinstance(v, dict):
                v = {a: b.copy() for a, b in v.items()}
            setattr(new, k, v)
        return new

    def factors(self, cls):
        """
        New ``EPFactors`` of class ``cls`` holding a copy of the factors
        """
        factors = cls(
            self.node_constraints,
            self.edge_parents,
            self.edge_children,
            self.block_nodes[ROOTWARD],
            self.block_nodes[LEAFWARD],
        )
        for k, v in self.fields.items():
            getattr(factors, k)[:] = v
        return factors

    def unscale(self, nodes, scale):
        """
        Make the state equivalent but with ``scale[nodes] = scale``, by dividing all
        factors associated with those nodes (see ``_rescale_factors``) by ``scale``.
        """
        sel = np.zeros(self.node_posterior.shape[0], dtype=bool)
        sel[nodes] = True
        f = self.fields
        f["edge"][sel[self.edge_parents], ROOTWARD] /= scale
        f["edge"][sel[self.edge_children], LEAFWARD] /= scale
        f["block"][sel[self.block_nodes[ROOTWARD]], ROOTWARD] /= scale
        f["block"][sel[self.block_nodes[LEAFWARD]], LEAFWARD] /= scale
        f["node"][sel, MIXPRIOR] /= scale
        f["node"][sel, CONSTRNT] /= scale
        f["scale"][sel] = scale

    def fix_nodes(self, nodes):
        """
        Constrain ``nodes`` to their true (simulated) ages
        """
        nodes = np.asarray(nodes)
        times = self.ts.nodes_time[nodes]
        self.node_constraints[nodes, LOWER] = times
        self.node_constraints[nodes, UPPER] = times

    @property
    def fixed(self):
        return self.node_constraints[:, LOWER] == self.node_constraints[:, UPPER]

    def branch_counts(self, unphased):
        """
        Number of edges (blocks) of each kind visited by the propagate functions
        """
        if unphased:
            p, c = self.block_nodes[ROOTWARD], self.block_nodes[LEAFWARD]
        else:
            p, c = self.edge_parents, self.edge_children
        fp, fc = self.fixed[p], self.fixed[c]
        return {
            "both_fixed": int(np.sum(fp & fc)),
            "fixed_parent": int(np.sum(fp & ~fc)),
            "fixed_child": int(np.sum(~fp & fc)),
            "twin": int(np.sum(~fp & ~fc & (p == c))),
            "both_free": int(np.sum(~fp & ~fc & (p != c))),
        }

    def after_pass(self, state):
        """
        Copy with the state returned by `run_likelihood` as its current state
        """
        posterior, _, *fields = state
        new = self.copy()
        new.node_posterior = posterior.copy()
        new.fields = {k: v.copy() for k, v in zip(self.FIELDS, fields)}
        return new


def make_snapshot(ts, name, iterations=2):
    """
    Snapshot of the state of EP on ``ts`` (fixture ``name``) after some iterations
    """
    singletons_phased, mutation_rate = SNAPSHOT_SPECS[name]
    ep = ExpectationPropagation(
        ts, mutation_rate=mutation_rate, singletons_phased=singletons_phased
    )
    for _ in range(iterations):
        ep.iterate(max_shape=1000, min_step=0.1)
    return Snapshot(ts, ep)


def fixed_internal_snapshot(snap, unphased, every=4):
    """
    Copy of a snapshot with every ``every``th internal node (every ``every``th block
    node, for unphased blocks) fixed at its true age; the first block has both nodes
    fixed. Gives all branches of the propagate functions.
    """
    snap = snap.copy()
    if unphased:
        nodes = np.unique(snap.block_nodes)
        fixed = list(nodes[::every])
        fixed += [snap.block_nodes[ROOTWARD][0], snap.block_nodes[LEAFWARD][0]]
    else:
        internal = np.flatnonzero(~snap.fixed)
        fixed = list(internal[::every])
    snap.fix_nodes(np.unique(fixed))
    return snap


# One edge (block) in each branch of `propagate_likelihood`/`propagate_mutations`:
# (unphased, constraints of the nodes, parent, child). Fixed nodes have ages 0
# (child) or 10 (parent) or 5 (block nodes).
INF = np.inf
BRANCHES = {
    "fixed_child": (False, [[0, 0], [0, INF]], 1, 0),
    "fixed_parent": (False, [[0, INF], [10, 10]], 1, 0),
    "both_free": (False, [[0, INF], [0, INF]], 1, 0),
    "both_fixed": (False, [[0, 0], [10, 10]], 1, 0),
    "twin": (True, [[0, INF], [0, INF]], 1, 1),
    "sideways_fixed_parent": (True, [[0, INF], [5, 5], [0, INF]], 1, 2),
    "sideways_fixed_child": (True, [[0, INF], [0, INF], [5, 5]], 1, 2),
    "unphased_both_free": (True, [[0, INF], [0, INF], [0, INF]], 1, 2),
    "unphased_both_fixed": (True, [[0, 0], [5, 5], [5, 5]], 1, 2),
}
# cavity and message of every node; `node` = cavity + message
CAVITY, MESSAGE = np.array([3.0, 0.5]), np.array([1.0, 0.1])
# message so large relative to the cavity that the cavity has to be damped
DAMPED_CAVITY, DAMPED_MESSAGE = np.array([1.0, 0.05]), np.array([22.0, 2.4])


def tiny_snapshot(branch, cavity=CAVITY, message=MESSAGE, likelihood=(3.0, 0.1)):
    """
    Snapshot of a single edge (or block) whose factor is `message` at each end
    """
    unphased, constraints, p, c = BRANCHES[branch]
    constraints = np.array(constraints, dtype=np.float64)
    n = constraints.shape[0]
    snap = Snapshot.__new__(Snapshot)
    snap.ts = None
    snap.node_constraints = constraints
    snap.node_posterior = np.tile(cavity + message, (n, 1))
    parents, children = np.array([p], np.int32), np.array([c], np.int32)
    snap.edge_parents, snap.edge_children = parents, children
    snap.block_nodes = np.array([parents, children])
    snap.edge_order = snap.block_order = np.zeros(1, np.int32)
    snap.edge_likelihoods = snap.block_likelihoods = np.array([likelihood])
    snap.edge_logconst = np.zeros(1)
    snap.block_logconst = np.zeros(1)
    snap.mutation_order = np.zeros(1, np.int32)
    snap.mutation_edges = np.zeros(1, np.int32)
    snap.mutation_blocks = np.full(1, 0 if unphased else tskit.NULL, np.int32)
    snap.num_mutations = 1
    fields = {
        "node": np.zeros((n, 2, 2)),
        "edge": np.zeros((1, 2, 2)),
        "block": np.zeros((1, 2, 2)),
        "scale": np.ones(n),
    }
    factor = fields["block" if unphased else "edge"]
    factor[0, ROOTWARD] = message
    factor[0, LEAFWARD] = message
    snap.fields = fields
    snap.unphased = unphased
    snap.unconstrained_roots = np.zeros(n, dtype=bool)
    return snap


def likelihood_args(snap, unphased, order=None, likelihoods=None):
    """
    Leading arguments of `propagate_likelihood` and the matching `lognorm`
    """
    if unphased:
        args = (
            snap.block_order if order is None else order,
            snap.block_nodes[ROOTWARD],
            snap.block_nodes[LEAFWARD],
            snap.block_likelihoods if likelihoods is None else likelihoods,
        )
        return args, snap.block_logconst
    args = (
        snap.edge_order if order is None else order,
        snap.edge_parents,
        snap.edge_children,
        snap.edge_likelihoods if likelihoods is None else likelihoods,
    )
    return args, snap.edge_logconst


def run_likelihood(
    ns, snap, unphased, max_shape=1000.0, min_step=0.1, order=None, likelihoods=None
):
    """
    Run `propagate_likelihood` of namespace ``ns`` on copies of ``snap``; return
    ``(posterior, lognorm, node, edge, block, scale)`` after the pass.
    """
    args, lognorm = likelihood_args(snap, unphased, order, likelihoods)
    factors = snap.factors(ns.EPFactors)
    posterior = snap.node_posterior.copy()
    lognorm = lognorm.copy()
    ns.propagate_likelihood(
        *args,
        snap.node_constraints,
        posterior,
        factors,
        lognorm,
        max_shape,
        min_step,
        USE_BLOCK_LIKELIHOOD if unphased else USE_EDGE_LIKELIHOOD,
    )
    return (posterior, lognorm, factors.node, factors.edge, factors.block, factors.scale)


def run_prior(ns, snap, free, max_shape=1000.0, em_maxitt=10, em_reltol=1e-8):
    """
    Run `propagate_prior`; return ``(posterior, node, scale)`` after the pass.
    """
    factors = snap.factors(ns.EPFactors)
    posterior = snap.node_posterior.copy()
    ns.propagate_prior(free, posterior, factors, max_shape, em_maxitt, em_reltol)
    return (posterior, factors.node, factors.scale)


def run_mutations(ns, snap, unphased, edges=None, posterior=None, order=None):
    """
    Run `propagate_mutations` on the phased (or unphased) mutations; return
    ``(mutation_posterior, phase, node, edge, block, scale)``.
    """
    factors = snap.factors(ns.EPFactors)
    mut_posterior = np.full((snap.num_mutations, 2), np.nan)
    phase = np.ones(snap.num_mutations)
    phased = snap.mutation_blocks == tskit.NULL
    if unphased:
        order = snap.mutation_order[~phased] if order is None else order
        args = (
            snap.mutation_blocks,
            snap.block_nodes[ROOTWARD],
            snap.block_nodes[LEAFWARD],
            snap.block_likelihoods,
        )
    else:
        order = snap.mutation_order[phased] if order is None else order
        args = (
            snap.mutation_edges if edges is None else edges,
            snap.edge_parents,
            snap.edge_children,
            snap.edge_likelihoods,
        )
    ns.propagate_mutations(
        order,
        mut_posterior,
        phase,
        *args,
        snap.node_constraints,
        snap.node_posterior if posterior is None else posterior,
        factors,
        USE_BLOCK_LIKELIHOOD if unphased else USE_EDGE_LIKELIHOOD,
    )
    return (
        mut_posterior,
        phase,
        factors.node,
        factors.edge,
        factors.block,
        factors.scale,
    )


def run_sweeps(ns, snap, num_sweeps, max_shape=1000.0, min_step=0.1, em_maxitt=10):
    """
    Run EP sweeps (the loop of `ExpectationPropagation.iterate`, then the mutation
    passes of `infer`) with namespace ``ns`` from a copy of ``snap``. Returns, per
    sweep, ``(posterior, edge_logconst, block_logconst)`` copies, and the mutation
    ``(posterior, phase)``.
    """
    factors = snap.factors(ns.EPFactors)
    posterior = snap.node_posterior.copy()
    edge_logconst = snap.edge_logconst.copy()
    block_logconst = snap.block_logconst.copy()
    mut_posterior = np.full((snap.num_mutations, 2), np.nan)
    mut_phase = np.ones(snap.num_mutations)
    block_nodes = snap.block_nodes
    states = []
    for _ in range(num_sweeps):
        ns.propagate_likelihood(
            snap.block_order,
            block_nodes[ROOTWARD],
            block_nodes[LEAFWARD],
            snap.block_likelihoods,
            snap.node_constraints,
            posterior,
            factors,
            block_logconst,
            max_shape,
            min_step,
            USE_BLOCK_LIKELIHOOD,
        )
        ns.propagate_likelihood(
            snap.edge_order,
            snap.edge_parents,
            snap.edge_children,
            snap.edge_likelihoods,
            snap.node_constraints,
            posterior,
            factors,
            edge_logconst,
            max_shape,
            min_step,
            USE_EDGE_LIKELIHOOD,
        )
        ns.propagate_prior(
            snap.unconstrained_roots, posterior, factors, max_shape, em_maxitt, 1e-8
        )
        ns._rescale_factors(factors)
        states.append((posterior.copy(), edge_logconst.copy(), block_logconst.copy()))
    phased = snap.mutation_blocks == tskit.NULL
    ns.propagate_mutations(
        snap.mutation_order[~phased],
        mut_posterior,
        mut_phase,
        snap.mutation_blocks,
        block_nodes[ROOTWARD],
        block_nodes[LEAFWARD],
        snap.block_likelihoods,
        snap.node_constraints,
        posterior,
        factors,
        USE_BLOCK_LIKELIHOOD,
    )
    ns.propagate_mutations(
        snap.mutation_order[phased],
        mut_posterior,
        mut_phase,
        snap.mutation_edges,
        snap.edge_parents,
        snap.edge_children,
        snap.edge_likelihoods,
        snap.node_constraints,
        posterior,
        factors,
        USE_EDGE_LIKELIHOOD,
    )
    return states, (mut_posterior, mut_phase)
