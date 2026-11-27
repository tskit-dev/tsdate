"""
Expectation propagation inputs for the equivalence tests of ``tsdate.variational``:
the inputs of ``ExpectationPropagation`` on the session fixtures (editable, so that
tests can force branches), and the loop of ``iterate``, run with the
``ExpectationPropagation`` of a given module (``_tsdate``, or the oracle
``oracle.variational``).
"""

import numpy as np
import tskit

from tsdate.variational import (
    LEAFWARD,
    LOWER,
    ROOTWARD,
    UPPER,
    USE_BLOCK_LIKELIHOOD,
    USE_EDGE_LIKELIHOOD,
    ExpectationPropagation,
)

# fixture name -> (singletons_phased, mutation_rate)
SPECS = {
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
    "unconstrained_roots",
    "roots",
    "leaves",
    "mutation_edges",
    "mutation_blocks",
)


class Inputs:
    """
    Copies of the inputs of an ``ExpectationPropagation``, which tests can edit
    """

    def __init__(self, ts, ep):
        self.ts = ts
        for name in _COPIED:
            setattr(self, name, getattr(ep, name).copy())

    def copy(self):
        new = object.__new__(Inputs)
        for k, v in self.__dict__.items():
            setattr(new, k, v.copy() if isinstance(v, np.ndarray) else v)
        return new

    def kwargs(self):
        """
        The arguments of ``ExpectationPropagation``
        """
        return dict(
            constraints=self.node_constraints,
            edges_parent=self.edge_parents,
            edges_child=self.edge_children,
            edges_likelihood=self.edge_likelihoods,
            blocks_left=self.block_nodes[ROOTWARD],
            blocks_right=self.block_nodes[LEAFWARD],
            blocks_likelihood=self.block_likelihoods,
        )

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
        Number of edges (blocks) of each kind visited by the likelihood passes
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

    def mutations_edge(self, unphased):
        """
        The edges (blocks) of the phased (unphased) mutations
        """
        phased = self.mutation_blocks == tskit.NULL
        if unphased:
            return self.mutation_blocks[~phased]
        return self.mutation_edges[phased]


def make_inputs(ts, name):
    """
    The inputs of EP on ``ts`` (fixture ``name``)
    """
    singletons_phased, mutation_rate = SPECS[name]
    ep = ExpectationPropagation(
        ts, mutation_rate=mutation_rate, singletons_phased=singletons_phased
    )
    return Inputs(ts, ep)


def fixed_internal(inputs, unphased, every=4):
    """
    Copy of the inputs with every ``every``th internal node (every ``every``th block
    node, for unphased blocks) fixed at its true age; the first block has both nodes
    fixed. Gives all branches of the passes.
    """
    inputs = inputs.copy()
    if unphased:
        nodes = np.unique(inputs.block_nodes)
        fixed = list(nodes[::every])
        fixed += [inputs.block_nodes[ROOTWARD][0], inputs.block_nodes[LEAFWARD][0]]
    else:
        internal = np.flatnonzero(~inputs.fixed)
        fixed = list(internal[::every])
    inputs.fix_nodes(np.unique(fixed))
    return inputs


def iterate(ep, inputs, max_shape=1000.0, min_step=0.1, em_maxitt=10, em_reltol=1e-8):
    """
    The loop of ``tsdate.variational.ExpectationPropagation.iterate``
    """
    ep.propagate_likelihood(
        order=inputs.block_order,
        unphased=USE_BLOCK_LIKELIHOOD,
        max_shape=max_shape,
        min_step=min_step,
    )
    ep.propagate_likelihood(
        order=inputs.edge_order,
        unphased=USE_EDGE_LIKELIHOOD,
        max_shape=max_shape,
        min_step=min_step,
    )
    ep.propagate_prior(
        free=inputs.unconstrained_roots,
        max_shape=max_shape,
        em_maxitt=em_maxitt,
        em_reltol=em_reltol,
    )
    ep.rescale_factors()


def start(module, inputs, iterations=2):
    """
    The ``ExpectationPropagation`` of ``module`` on ``inputs`` after ``iterations``
    """
    ep = module.ExpectationPropagation(**inputs.kwargs())
    for _ in range(iterations):
        iterate(ep, inputs)
    return ep


def state(ep):
    """
    Copies of the node posteriors and the log normalising constants
    """
    return ep.posterior, ep.edges_lognorm, ep.blocks_lognorm
