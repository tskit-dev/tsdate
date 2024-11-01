# MIT License
#
# Copyright (c) 2021-25 Tskit Developers
# Copyright (c) 2020-21 University of Oxford
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Test cases for dating noncontemporary samples
"""

import msprime
import numpy as np
import pytest
import tskit

from tsdate import date


def simulate_internal_samples(pop_size=2, sample_time=6, L=4, r=0.1, random_seed=1):
    """
    Simulate ts with noncontemporary samples that have descendants
    https://github.com/tskit-dev/msprime/discussions/2260
    """
    rng = np.random.default_rng(random_seed)

    samples = [
        msprime.SampleSet(pop_size),
        msprime.SampleSet(1, ploidy=1, time=sample_time + 1),
    ]

    # recent history up to sample time
    ts1 = msprime.sim_ancestry(
        samples=samples,
        population_size=pop_size,
        model="dtwf",
        end_time=sample_time,
        sequence_length=L,
        recombination_rate=r,
        random_seed=random_seed,
    )

    # remove dummy node above the sample time
    ts1 = ts1.simplify(range(2 * pop_size), keep_unary=True)

    # history before sample time
    ts2 = msprime.sim_ancestry(
        samples=[msprime.SampleSet(pop_size, time=sample_time)],
        model="dtwf",
        population_size=pop_size,
        sequence_length=L,
        recombination_rate=r,
        random_seed=random_seed + 1000,
    )

    # remap roots to samples in ts2
    roots = [n.id for n in ts1.nodes() if n.time == sample_time]
    tips = ts2.samples()
    rng.shuffle(tips)
    node_mapping = [tskit.NULL for _ in ts1.nodes()]
    for t, n in zip(tips[: len(roots)], roots):
        node_mapping[n] = t
    ts = ts2.union(ts1, node_mapping, check_shared_equality=False)

    return ts


def get_internal_samples(ts):
    always_internal = np.full(ts.num_nodes, False)
    always_internal[list(ts.samples())] = True
    for t in ts.trees():
        for n in t.samples():
            if t.num_children(n) == 0:
                always_internal[n] = False
    return np.flatnonzero(always_internal)


class TestInternalConstraints:
    """
    Test dating when some internal nodes are fixed

    NB: Currently we need to use `constr_iterations` to
    ensure that the constraint forcing doesn't modify sample
    ages
    """

    def test_with_unary_internal_samples(self):
        mu = 0.1
        ts = simulate_internal_samples(random_seed=1).simplify()
        ts = msprime.sim_mutations(ts, rate=mu, random_seed=1)
        dts = date(ts, mutation_rate=mu, rescaling_intervals=1)
        assert np.allclose(
            dts.nodes_time[list(dts.samples())],
            ts.nodes_time[list(ts.samples())],
        )

    def test_with_nonunary_internal_samples(self):
        rng = np.random.default_rng(1)
        ts = msprime.sim_ancestry(
            samples=10,
            population_size=1e4,
            sequence_length=1e5,
            recombination_rate=1e-8,
            random_seed=1,
        )
        ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=1)
        fixed_nodes = np.unique(rng.integers(ts.num_samples, ts.num_nodes, size=10))
        nodes_flags = ts.nodes_flags.copy()
        nodes_flags[fixed_nodes] = tskit.NODE_IS_SAMPLE
        tab = ts.dump_tables()
        tab.nodes.flags = nodes_flags
        tab.sort()
        ts = tab.tree_sequence()
        dts = date(ts, mutation_rate=1e-8, rescaling_intervals=1)
        assert np.allclose(
            dts.nodes_time[fixed_nodes],
            ts.nodes_time[fixed_nodes],
        )

    def test_that_unary_nonsample_fails(self):
        mu = 0.1
        ts = simulate_internal_samples(random_seed=1).simplify()
        ts = msprime.sim_mutations(ts, rate=mu, random_seed=1)
        internal_samples = get_internal_samples(ts)
        tab = ts.dump_tables()
        new_flags = tab.nodes.flags.copy()
        new_flags[internal_samples] = 0
        tab.nodes.flags = new_flags
        ts_no_internal = tab.tree_sequence()
        with pytest.raises(ValueError):
            date(
                ts_no_internal,
                mutation_rate=mu,
            )


class TestUnconstrainedSamples:
    """
    Test dating when some leaf nodes do not have a fixed time
    """

    def test_with_free_leaf_nodes(self):
        ts = msprime.sim_ancestry(
            samples=10,
            population_size=1e4,
            sequence_length=1e5,
            recombination_rate=1e-8,
            random_seed=1,
        )
        ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=1)
        free_nodes = np.array([1, 3, 18]).astype(np.int32)
        nodes_flags = ts.nodes_flags.copy()
        nodes_flags[free_nodes] = 0
        tab = ts.dump_tables()
        tab.nodes.flags = nodes_flags
        tab.sort()
        ts = tab.tree_sequence()
        dts = date(ts, mutation_rate=1e-8, rescaling_intervals=1)
        assert np.all(ts.nodes_time[free_nodes] == 0.0)
        assert np.all(dts.nodes_time[free_nodes] > 0)
