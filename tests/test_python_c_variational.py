"""
Tests of the expectation propagation functions of the low-level _tsdate module that
the oracle tests in test_oracle_variational.py do not reach: argument handling, and
the messages of library errors.
"""

import _tsdate
import numpy as np
import pytest

INF = np.inf


def factors(num_nodes=3, num_edges=2, num_blocks=1):
    """
    The tuple of arrays that describes EP factors, for a small example.
    """
    return (
        np.zeros((num_nodes, 2, 2)),
        np.zeros((num_edges, 2, 2)),
        np.zeros((num_blocks, 2, 2)),
        np.ones(num_nodes),
        np.array([2, 2][:num_edges], dtype=np.int32),
        np.array([0, 1][:num_edges], dtype=np.int32),
        np.full(num_blocks, 2, dtype=np.int32),
        np.full(num_blocks, 1, dtype=np.int32),
    )


def likelihood_args(**changes):
    """
    Arguments of propagate_likelihood on two edges (2, 0) and (2, 1).
    """
    args = dict(
        edge_order=np.array([0, 1], dtype=np.int32),
        edges_parent=np.array([2, 2], dtype=np.int32),
        edges_child=np.array([0, 1], dtype=np.int32),
        likelihoods=np.array([[1.0, 0.1], [1.0, 0.1]]),
        constraints=np.array([[0.0, 0.0], [0.0, INF], [0.0, INF]]),
        posterior=np.ones((3, 2)),
        factors=factors(),
        lognorm=np.zeros(2),
        max_shape=1000.0,
        min_step=0.1,
        unphased=False,
    )
    args.update(changes)
    return list(args.values())


def prior_args(**changes):
    args = dict(
        free=np.array([False, True, True]),
        posterior=np.full((3, 2), 2.0),
        factors=factors(),
        max_shape=1000.0,
        em_maxitt=10,
        em_reltol=1e-8,
    )
    args.update(changes)
    return list(args.values())


def mutation_args(**changes):
    args = dict(
        mutations_order=np.array([0, 1], dtype=np.int32),
        mutations_posterior=np.zeros((2, 2)),
        mutations_phase=np.zeros(2),
        mutations_edge=np.array([0, 1], dtype=np.int32),
        edges_parent=np.array([2, 2], dtype=np.int32),
        edges_child=np.array([0, 1], dtype=np.int32),
        likelihoods=np.array([[1.0, 0.1], [1.0, 0.1]]),
        constraints=np.array([[1.0, 1.0], [0.0, INF], [0.0, INF]]),
        posterior=np.ones((3, 2)),
        factors=factors(),
        unphased=False,
    )
    args.update(changes)
    return list(args.values())


class TestFactors:
    def test_not_a_tuple(self):
        with pytest.raises(TypeError, match="tuple of eight"):
            _tsdate.rescale_factors(list(factors()))
        with pytest.raises(TypeError, match="tuple of eight"):
            _tsdate.assemble_factors(factors()[:7])

    def test_wrong_dtype(self):
        f = list(factors())
        f[4] = f[4].astype(np.int64)
        with pytest.raises(TypeError):
            _tsdate.rescale_factors(tuple(f))

    def test_wrong_shape(self):
        f = list(factors())
        f[1] = np.zeros((2, 2, 3))
        with pytest.raises(ValueError, match="shape"):
            _tsdate.rescale_factors(tuple(f))

    def test_inconsistent_sizes(self):
        f = list(factors())
        f[3] = np.ones(4)
        with pytest.raises(ValueError, match="inconsistent"):
            _tsdate.assemble_factors(tuple(f))

    def test_readonly_arrays(self):
        f = list(factors())
        f[0].flags.writeable = False
        with pytest.raises(ValueError):
            _tsdate.rescale_factors(tuple(f))
        # nothing is written by the assembly
        assert _tsdate.assemble_factors(tuple(f)).shape == (3, 2)

    def test_rescale_in_place(self):
        f = factors()
        f[3][:] = [2.0, 3.0, 4.0]
        f[0][:] = 1.0
        _tsdate.rescale_factors(f)
        assert np.all(f[3] == 1.0)
        assert np.array_equal(f[0][:, 0, 0], [2.0, 3.0, 4.0])


class TestCheckValidConstraints:
    def test_columns(self):
        with pytest.raises(ValueError, match="two columns"):
            _tsdate.check_valid_constraints(
                np.zeros((3, 3)), np.zeros(0, np.int32), np.zeros(0, np.int32)
            )

    def test_edge_sizes(self):
        with pytest.raises(ValueError, match="same size"):
            _tsdate.check_valid_constraints(
                np.zeros((3, 2)), np.zeros(2, np.int32), np.zeros(1, np.int32)
            )


class TestPropagateLikelihood:
    def test_runs(self):
        args = likelihood_args()
        assert _tsdate.propagate_likelihood(*args) is None
        assert np.all(np.isfinite(args[7]))

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ("edges_child", "same size"),
            ("likelihoods", "shape"),
            ("constraints", "shape"),
            ("posterior", "same shape"),
            ("lognorm", "one entry per edge"),
        ],
    )
    def test_shape_errors(self, change, message):
        value = dict(
            edges_child=np.zeros(1, np.int32),
            likelihoods=np.zeros((1, 2)),
            constraints=np.zeros((2, 2)),
            posterior=np.ones((2, 2)),
            lognorm=np.zeros(3),
        )[change]
        with pytest.raises(ValueError, match=message):
            _tsdate.propagate_likelihood(*likelihood_args(**{change: value}))

    def test_factors_do_not_match_kind(self):
        # the unphased pass uses the blocks, of which there is one
        with pytest.raises(ValueError, match="one entry per edge"):
            _tsdate.propagate_likelihood(*likelihood_args(unphased=True))

    def test_library_errors(self):
        with pytest.raises(_tsdate.LibraryError, match="at least one"):
            _tsdate.propagate_likelihood(*likelihood_args(max_shape=0.5))
        with pytest.raises(_tsdate.LibraryError, match="greater than zero and less"):
            _tsdate.propagate_likelihood(*likelihood_args(min_step=1.0))
        with pytest.raises(_tsdate.LibraryError, match="not greater than -1"):
            _tsdate.propagate_likelihood(
                *likelihood_args(posterior=np.full((3, 2), -5.0))
            )

    @pytest.mark.parametrize("kind", ["float32", "strided"])
    def test_array_updated_in_place_must_not_be_converted(self, kind):
        # a copy would silently lose the update
        if kind == "float32":
            posterior = np.ones((3, 2), dtype=np.float32)
        else:
            posterior = np.ones((3, 4))[:, ::2]
        with pytest.raises(ValueError, match="writeable, C-contiguous"):
            _tsdate.propagate_likelihood(*likelihood_args(posterior=posterior))

    @pytest.mark.parametrize("improper", [0, 1, 2])
    def test_improper_posterior_of_free_edge(self, improper):
        # the posterior of either node of an edge (or of a twin block) is not proper
        posterior = np.full((3, 2), 2.0)
        posterior[improper] = -5.0
        args = likelihood_args(
            constraints=np.array([[0.0, INF]] * 3),
            posterior=posterior,
            edges_parent=np.array([0, 2], dtype=np.int32),
            edges_child=np.array([1, 2], dtype=np.int32),
        )
        with pytest.raises(_tsdate.LibraryError, match="not greater than -1"):
            _tsdate.propagate_likelihood(*args)

    def test_improper_posterior_of_twin_block(self):
        args = likelihood_args(
            constraints=np.array([[0.0, INF]] * 3),
            posterior=np.array([[2.0, 2.0], [2.0, 2.0], [-5.0, 2.0]]),
            edges_parent=np.array([2, 2], dtype=np.int32),
            edges_child=np.array([2, 2], dtype=np.int32),
            factors=factors(num_blocks=2),
            unphased=True,
        )
        with pytest.raises(_tsdate.LibraryError, match="not greater than -1"):
            _tsdate.propagate_likelihood(*args)

    def test_twin_projection_fails(self):
        # a zero cavity (the damping step is one for zeros) and no mutation target
        args = likelihood_args(
            constraints=np.array([[0.0, INF]] * 3),
            posterior=np.zeros((3, 2)),
            edges_parent=np.array([2, 2], dtype=np.int32),
            edges_child=np.array([2, 2], dtype=np.int32),
            likelihoods=np.array([[1.0, 0.0], [1.0, 0.0]]),
            factors=factors(num_blocks=2),
            unphased=True,
        )
        with pytest.raises(_tsdate.LibraryError, match="division by zero"):
            _tsdate.propagate_likelihood(*args)

    def test_wrong_dtype(self):
        with pytest.raises(TypeError):
            _tsdate.propagate_likelihood(
                *likelihood_args(edge_order=np.array([0, 1], dtype=np.int64))
            )


class TestPropagatePrior:
    def test_runs(self):
        args = prior_args()
        assert _tsdate.propagate_prior(*args) is None
        assert np.all(args[1][1:, 1] > 2.0)

    def test_sizes(self):
        with pytest.raises(ValueError, match="one entry per node"):
            _tsdate.propagate_prior(*prior_args(free=np.ones(2, dtype=bool)))
        with pytest.raises(ValueError, match="one entry per node"):
            _tsdate.propagate_prior(*prior_args(posterior=np.ones((2, 2))))

    def test_library_errors(self):
        with pytest.raises(_tsdate.LibraryError, match="at least one"):
            _tsdate.propagate_prior(*prior_args(max_shape=0.0))
        with pytest.raises(_tsdate.LibraryError, match="penalty is not positive"):
            _tsdate.propagate_prior(*prior_args(posterior=np.zeros((3, 2))))

    def test_improper_posterior_cannot_be_rescaled(self):
        # the fit succeeds (the other node has a large shape) but one posterior is
        # not proper, so that it cannot be rescaled
        posterior = np.array([[2.0, 2.0], [-5.0, 1.0], [99.0, 1.0]])
        with pytest.raises(_tsdate.LibraryError, match="not greater than -1"):
            _tsdate.propagate_prior(*prior_args(posterior=posterior))


class TestPropagateMutations:
    def test_runs(self):
        args = mutation_args()
        assert _tsdate.propagate_mutations(*args) is None
        assert np.all(np.isfinite(args[2]))

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ("mutations_edge", "same size"),
            ("mutations_posterior", "shape"),
            ("edges_child", "same size"),
            ("likelihoods", "shape"),
            ("constraints", "shape"),
            ("posterior", "same shape"),
        ],
    )
    def test_shape_errors(self, change, message):
        value = dict(
            mutations_edge=np.zeros(1, np.int32),
            mutations_posterior=np.zeros((3, 2)),
            edges_child=np.zeros(1, np.int32),
            likelihoods=np.zeros((1, 2)),
            constraints=np.zeros((2, 2)),
            posterior=np.ones((2, 2)),
        )[change]
        with pytest.raises(ValueError, match=message):
            _tsdate.propagate_mutations(*mutation_args(**{change: value}))

    def test_factors_do_not_match_kind(self):
        with pytest.raises(ValueError, match="one entry per edge"):
            _tsdate.propagate_mutations(*mutation_args(unphased=True))


class TestArgumentParsing:
    @pytest.mark.parametrize(
        "name",
        [
            "rescale_factors",
            "assemble_factors",
            "check_valid_constraints",
            "propagate_likelihood",
            "propagate_prior",
            "propagate_mutations",
        ],
    )
    def test_wrong_number_of_arguments(self, name):
        with pytest.raises(TypeError):
            getattr(_tsdate, name)()

    def test_check_valid_constraints_wrong_dtype(self):
        with pytest.raises(TypeError):
            _tsdate.check_valid_constraints(
                np.zeros((3, 2)), np.zeros(0, np.int64), np.zeros(0, np.int32)
            )

    def test_bad_factors(self):
        bad = list(factors())
        for func, args in [
            (_tsdate.propagate_likelihood, likelihood_args(factors=bad)),
            (_tsdate.propagate_prior, prior_args(factors=bad)),
            (_tsdate.propagate_mutations, mutation_args(factors=bad)),
        ]:
            with pytest.raises(TypeError, match="tuple of eight"):
                func(*args)

    def test_wrong_dtypes(self):
        for func, args in [
            (_tsdate.propagate_prior, prior_args(free=np.ones(3))),
            (
                _tsdate.propagate_mutations,
                mutation_args(mutations_order=np.zeros(1, np.int64)),
            ),
        ]:
            with pytest.raises(TypeError):
                func(*args)
