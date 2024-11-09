# MIT License
#
# Copyright (c) 2021-24 Tskit Developers
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
Infer the age of nodes from mutational data, conditional on a tree sequence topology.
"""

import logging
import time  # DEBUG
from collections import namedtuple

import numpy as np
import tskit

from . import demography, discrete, prior, provenance, schemas, util, variational
from .node_time_class import LIN_GRID, LOG_GRID

logger = logging.getLogger(__name__)

FORMAT_NAME = "tsdate"
DEFAULT_RESCALING_INTERVALS = 1000
DEFAULT_RESCALING_ITERATIONS = 5
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_EPSILON = 1e-6


# Classes for each method
Results = namedtuple(
    "Results",
    [
        "posterior_mean",
        "posterior_var",
        "mutation_mean",
        "mutation_var",
        "mutation_lik",
        "mutation_edge",
        "mutation_node",
        "method_object",
    ],
)


class EstimationMethod:
    """
    Base class to hold the various estimation methods. Override prior_grid_func_name with
    something like "parameter_grid" or "prior_grid".
    """

    prior_grid_func_name = None

    def run():
        # Subclasses should override to return a return a Results object
        raise NotImplementedError(
            "Base class 'EstimationMethod' not intended for direct use"
        )

    def __init__(
        self,
        ts,
        *,
        mutation_rate=None,
        population_size=None,
        recombination_rate=None,
        time_units=None,
        priors=None,
        return_likelihood=None,
        return_model=None,
        record_provenance=None,
        constr_iterations=None,
        progress=None,
        # Deprecated params
        return_posteriors=None,
    ):
        # Set up all the generic params describe in the tsdate.date function, and define
        # priors if not passed-in already
        if return_posteriors is not None:
            raise ValueError(
                'The "return_posteriors" parameter has been deprecated. Either use the '
                "posterior values encoded in node metadata or set ``return_model=True`` "
                "then access `model.node_posteriors()` to obtain a transposed version "
                "of the matrix previously returned when ``return_posteriors=True.``"
            )
        self.ts = ts
        self.mutation_rate = mutation_rate
        self.recombination_rate = recombination_rate
        self.return_model = return_model
        self.return_likelihood = return_likelihood
        self.pbar = progress
        self.time_units = "generations" if time_units is None else time_units
        if record_provenance is None:
            record_provenance = True

        if recombination_rate is not None:
            raise NotImplementedError(
                "Using the recombination clock is not currently supported"
                ". See https://github.com/awohns/tsdate/issues/5 for details"
            )

        Ne = population_size  # shorthand
        if isinstance(Ne, dict):
            Ne = demography.PopulationSizeHistory(**Ne)

        self.provenance_params = None
        if record_provenance:
            self.provenance_params = dict(
                mutation_rate=mutation_rate,
                recombination_rate=recombination_rate,
                time_units=time_units,
                progress=progress,
                # demography.PopulationSizeHistory provides as_dict() for saving
                population_size=Ne.as_dict() if hasattr(Ne, "as_dict") else Ne,
            )

        if constr_iterations is None:
            self.constr_iterations = 0
        else:
            if not (isinstance(constr_iterations, int) and constr_iterations >= 0):
                raise ValueError(
                    "Number of constrained least squares iterations must be a "
                    "non-negative integer"
                )
            self.constr_iterations = constr_iterations

        if self.prior_grid_func_name is None:
            if priors is not None:
                raise ValueError(f"Priors are not used for method {self.name}")
            if Ne is not None:
                raise ValueError(f"Population size is not used for method {self.name}")
        else:
            if priors is None:
                if Ne is None:
                    raise ValueError(
                        "Must specify population size if priors are not already "
                        f"built using tsdate.build_{self.prior_grid_func_name}()"
                    )
                mk_prior = getattr(prior, self.prior_grid_func_name)
                # Default to not creating approximate priors unless ts has
                # greater than DEFAULT_APPROX_PRIOR_SIZE samples
                approx = ts.num_samples > prior.DEFAULT_APPROX_PRIOR_SIZE
                self.priors = mk_prior(
                    ts, Ne, approximate_priors=approx, progress=progress
                )
            else:
                logger.info("Using user-specified priors")
                if Ne is not None:
                    raise ValueError(
                        "Cannot specify population size if specifying priors "
                        f"from tsdate.build_{self.prior_grid_func_name}()"
                    )
                self.priors = priors

        # mutation to edge mapping
        # TODO: this isn't needed except for mutations_edge in constrain_mutations
        self.edges_mutations, self.mutations_edge = util.mutation_span_array(ts)

    def get_modified_ts(self, result, eps):
        # Return a new ts based on the existing one, but with the various
        # time-related information correctly set.
        ts = self.ts
        node_mean_t = result.posterior_mean
        node_var_t = result.posterior_var
        mut_mean_t = result.mutation_mean
        mut_var_t = result.mutation_var
        mut_edge = result.mutation_edge
        mut_node = result.mutation_node
        tables = ts.dump_tables()
        nodes = tables.nodes
        mutations = tables.mutations

        if self.provenance_params is not None:
            provenance.record_provenance(tables, self.name, **self.provenance_params)
        # Constrain node ages for positive branch lengths
        constr_timing = time.time()
        nodes.time = util.constrain_ages(ts, node_mean_t, eps, self.constr_iterations)
        mutations.time = util.constrain_mutations(ts, nodes.time, mut_edge)
        mutations.node = mut_node
        mutations.parent = np.full(mutations.num_rows, tskit.NULL, dtype=np.int32)
        tables.time_units = self.time_units
        constr_timing -= time.time()
        logger.info(f"Constrained node ages in {abs(constr_timing):.2f} seconds")
        # Add posterior mean and variance to node/mutation metadata
        meta_timing = time.time()
        self.set_time_metadata(
            nodes, node_mean_t, node_var_t, schemas.default_node_schema, overwrite=True
        )
        self.set_time_metadata(
            mutations, mut_mean_t, mut_var_t, schemas.default_mutation_schema
        )
        meta_timing -= time.time()
        logger.info(f"Inserted node and mutation metadata in {abs(meta_timing)} seconds")
        sort_timing = time.time()
        tables.sort()
        tables.build_index()
        tables.compute_mutation_parents()
        sort_timing -= time.time()
        logger.info(f"Sorted tree sequence in {abs(sort_timing):.2f} seconds")
        return tables.tree_sequence()

    def set_time_metadata(self, table, mean, var, default_schema, overwrite=False):
        if var is not None:
            table_name = type(table).__name__
            assert len(mean) == len(var) == table.num_rows
            if table.metadata_schema.schema is None or overwrite:
                if len(table.metadata) == 0 or overwrite:
                    table.metadata_schema = default_schema
                    md_iter = ({} for _ in range(table.num_rows))
                    # For speed, assume we don't need to validate
                    encoder = table.metadata_schema.encode_row
                    logger.info(f"Set metadata schema on {table_name}")
                else:
                    logger.warning(
                        f"Could not set metadata on {table_name}: "
                        "data already exists with no schema"
                    )
                    return
            else:
                md_iter = (
                    table.metadata_schema.decode_row(md)
                    for md in tskit.unpack_bytes(table.metadata, table.metadata_offset)
                )
                encoder = table.metadata_schema.validate_and_encode_row
                # TODO: could try to add to the existing schema if it's compatible
            metadata_array = []
            try:
                # wrap entire loop in try/except so metadata is either all set or not
                for metadata_dict, mn, vr in zip(md_iter, mean, var):
                    metadata_dict.update((("mn", mn), ("vr", vr)))
                    # validate and replace
                    metadata_array.append(encoder(metadata_dict))
                table.packset_metadata(metadata_array)
            except tskit.MetadataValidationError as e:
                logger.warning(f"Could not set time metadata in {table_name}: {e}")

    def parse_result(self, result, epsilon):
        # Construct the tree sequence to return and add other stuff we might want to
        # return. pst_cols is a dict to be appended to the output posterior dict
        ret = [self.get_modified_ts(result, epsilon)]
        if self.return_model:
            ret.append(result.method_object)
        if self.return_likelihood:
            ret.append(result.mutation_lik)
        return tuple(ret) if len(ret) > 1 else ret.pop()

    def get_fixed_nodes_set(self):
        # TODO: modify to allow non-contemporary samples. If these have priors specified
        # they should work fine with these algorithms.
        for sample in self.ts.samples():
            if self.ts.node(sample).time != 0:
                raise NotImplementedError("Samples must all be at time 0")
        return set(self.ts.samples())


class DiscreteTimeMethod(EstimationMethod):
    prior_grid_func_name = "prior_grid"

    @staticmethod
    def mean_var(ts, posterior):
        """
        Mean and variance of node age given an atomic time discretization. Fixed
        nodes will be given a mean of their exact time in the tree sequence, and
        zero variance. This is a static method for ease of testing.
        """
        mn_post = np.full(ts.num_nodes, np.nan)  # Fill with NaNs so we detect when
        va_post = np.full(ts.num_nodes, np.nan)  # there's been an error

        is_fixed = np.ones(posterior.num_nodes, dtype=bool)
        is_fixed[posterior.nonfixed_nodes] = False
        mn_post[is_fixed] = ts.nodes_time[is_fixed]
        va_post[is_fixed] = 0

        for u in posterior.nonfixed_nodes:
            probs = posterior[u]
            times = posterior.timepoints
            mn_post[u] = np.sum(probs * times) / np.sum(probs)
            va_post[u] = np.sum(((mn_post[u] - (times)) ** 2) * (probs / np.sum(probs)))

        return mn_post, va_post

    def main_algorithm(self, probability_space, epsilon, num_threads):
        # Algorithm class is shared by inside-outside & outside-maximization methods
        if probability_space == LIN_GRID:
            liklhd = discrete.Likelihoods(
                self.ts,
                self.priors.timepoints,
                self.mutation_rate,
                self.recombination_rate,
                eps=epsilon,
                fixed_node_set=self.get_fixed_nodes_set(),
                progress=self.pbar,
            )
        elif probability_space == LOG_GRID:
            liklhd = discrete.LogLikelihoods(
                self.ts,
                self.priors.timepoints,
                self.mutation_rate,
                self.recombination_rate,
                eps=epsilon,
                fixed_node_set=self.get_fixed_nodes_set(),
                progress=self.pbar,
            )
        else:
            raise ValueError(
                f"Invalid discrete probability space: {probability_space}. Must be "
                f"one of {LIN_GRID} or {LOG_GRID}"
            )
        if self.mutation_rate is not None:
            liklhd.precalculate_mutation_likelihoods(num_threads=num_threads)

        return discrete.InOutModel(self.priors, liklhd, progress=self.pbar)


class InsideOutsideMethod(DiscreteTimeMethod):
    name = "inside_outside"

    def run(
        self,
        eps,
        outside_standardize,
        ignore_oldest_root,
        probability_space,
        num_threads=None,
        cache_inside=None,
    ):
        if self.mutation_rate is None and self.recombination_rate is None:
            if self.ts.num_trees > 1:
                raise NotImplementedError(
                    "Specifying no mutation or recombination rate implies dating using "
                    "the topology-only clock. This produces biased results under "
                    "recombination (https://github.com/tskit-dev/tsdate/issues/292). "
                    "The topology-only clock has therefore been deprecated for tree "
                    "sequences representing more than one tree."
                )
        if self.provenance_params is not None:
            self.provenance_params.update(
                {k: v for k, v in locals().items() if k != "self"}
            )
        model = self.main_algorithm(probability_space, eps, num_threads)
        marginal_likl = model.inside_pass(cache_inside=cache_inside)
        model.outside_pass(
            standardize=outside_standardize, ignore_oldest_root=ignore_oldest_root
        )
        # Turn the posterior into probabilities
        model.posterior_grid.standardize()  # Just to ensure no floating point issues
        model.posterior_grid.force_probability_space(LIN_GRID)
        model.posterior_grid.to_probabilities()

        posterior_mean, posterior_var = self.mean_var(self.ts, model.posterior_grid)
        mut_edge = np.full(self.ts.num_mutations, tskit.NULL)
        mut_node = self.ts.mutations_node
        return Results(
            posterior_mean,
            posterior_var,
            None,
            None,
            marginal_likl,
            mut_edge,
            mut_node,
            model,
        )


class MaximizationMethod(DiscreteTimeMethod):
    name = "maximization"

    def __init__(self, ts, **kwargs):
        super().__init__(ts, **kwargs)

    def run(
        self,
        eps,
        probability_space=None,
        num_threads=None,
        cache_inside=None,
    ):
        if self.mutation_rate is None and self.recombination_rate is None:
            raise ValueError("Outside maximization method requires mutation rate")
        if self.provenance_params is not None:
            self.provenance_params.update(
                {k: v for k, v in locals().items() if k != "self"}
            )
        model = self.main_algorithm(probability_space, eps, num_threads)
        marginal_likl = model.inside_pass(cache_inside=cache_inside)
        model.outside_maximization(eps=eps)
        mut_edge = np.full(self.ts.num_mutations, tskit.NULL)
        mut_node = self.ts.mutations_node
        return Results(
            model.posterior_mean,
            None,
            None,
            None,
            marginal_likl,
            mut_edge,
            mut_node,
            model,
        )


class VariationalGammaMethod(EstimationMethod):
    prior_grid_func_name = None
    name = "variational_gamma"

    def __init__(self, ts, **kwargs):
        super().__init__(ts, **kwargs)

    def run(
        self,
        eps,
        max_iterations,
        max_shape,
        rescaling_intervals,
        rescaling_iterations,
        match_segregating_sites,
        regularise_roots,
        singletons_phased,
    ):
        if self.provenance_params is not None:
            self.provenance_params.update(
                {k: v for k, v in locals().items() if k != "self"}
            )
        if not max_iterations > 0:
            raise ValueError("Maximum number of EP iterations must be greater than 0")
        if self.mutation_rate is None:
            raise ValueError("Variational gamma method requires mutation rate")

        model = variational.ExpectationPropagationModel(
            self.ts,
            mutation_rate=self.mutation_rate,
            singletons_phased=singletons_phased,
        )
        model.infer(
            ep_iterations=max_iterations,
            max_shape=max_shape,
            rescale_intervals=rescaling_intervals,
            rescale_iterations=rescaling_iterations,
            regularise=regularise_roots,
            rescale_segsites=match_segregating_sites,
            progress=self.pbar,
        )
        marginal_likl = model.marginal_likelihood()
        node_mn, node_va = model.node_moments()
        mutation_mn, mutation_va = model.mutation_moments()
        mutation_edge, mutation_node = model.mutation_mapping()

        return Results(
            node_mn,
            node_va,
            mutation_mn,
            mutation_va,
            marginal_likl,
            mutation_edge,
            mutation_node,
            model,
        )


def maximization(
    tree_sequence,
    *,
    mutation_rate,
    population_size=None,
    priors=None,
    eps=None,
    num_threads=None,
    probability_space=None,
    # below deliberately undocumented
    cache_inside=None,
    Ne=None,
    # Other params documented in `.date()`
    **kwargs,
):
    """
    maximization(tree_sequence, *, mutation_rate, population_size=None, priors=None,\
        eps=None, num_threads=None, probability_space=None, **kwargs)

    Infer dates for nodes in a genealogical graph using the "outside maximization"
    algorithm. This approximates the marginal posterior distribution of a node's
    age using an atomic discretization of time (e.g. point masses at particular
    timepoints).

    This estimation method comprises a single "inside" step followed by an
    "outside maximization" step. The inside step passes backwards in time from the
    samples to the roots of the graph,taking account of the distributions of times of
    each node's child (and if a ``mutation_rate`` is given, the the number of mutations
    on each edge). The outside maximization step passes forwards in time from the roots,
    updating each node's time on the basis of the most likely timepoint for
    each parent of that node. This provides a reasonable point estimate for node times,
    but does not generate a true posterior time distribution.

    For example:

    .. code-block:: python

      new_ts = tsdate.maximization(ts, mutation_rate=1e-8, population_size=1e4)

    .. note::
        The prior parameters for each node-to-be-dated take the form of probabilities
        for each node at a set of discrete timepoints. If the ``priors`` parameter is
        used, it must specify an object constructed using :func:`build_prior_grid`
        (this can be used to define the number and position of the timepoints).
        If ``priors`` is not used, ``population_size`` must be provided,
        which is used to create a default prior derived from the conditional coalescent
        (tilted according to population size and weighted by the genomic
        span over which a node has a given number of descendant samples). This default
        prior assumes the nodes to be dated are all the non-sample nodes in the input
        tree sequence, and that they are contemporaneous.

    :param ~tskit.TreeSequence tree_sequence: The input tree sequence to be dated.
    :param float mutation_rate: The estimated mutation rate per unit of genome per
        unit time. If provided, the dating algorithm will use a mutation rate clock to
        help estimate node dates. Default: ``None``
    :param float or ~demography.PopulationSizeHistory population_size: The estimated
        (diploid) effective population size used to construct the (default) conditional
        coalescent prior. For a population with constant size, this can be given as a
        single value (for example, as commonly estimated by the observed genetic
        diversity of the sample divided by four-times the expected mutation rate).
        Alternatively, for a population with time-varying size, this can be given
        directly as a :class:`~demography.PopulationSizeHistory` object or a parameter
        dictionary passed to initialise a :class:`~demography.PopulationSizeHistory`
        object. The ``population_size`` parameter is only used when ``priors`` is
        ``None``. Conversely, if ``priors`` is not ``None``, no ``population_size``
        value should be specified.
    :param tsdate.node_time_class.NodeTimeValues priors: NodeTimeValues object containing
        the prior parameters for each node-to-be-dated. Note that different estimation
        methods may require different types of prior, as described in the documentation
        for each estimation method.
    :param float eps: The error factor in time difference calculations, and the
        minimum distance separating parent and child ages in the returned tree sequence.
        Default: None, treated as 1e-6.
    :param int num_threads: The number of threads to use when precalculating likelihoods.
        A simpler unthreaded algorithm is used unless this is >= 1. Default: None
    :param string probability_space: Should the internal algorithm save
        probabilities in "logarithmic" (slower, less liable to to overflow) or
        "linear" space (fast, may overflow). Default: None treated as"logarithmic"
    :param \\**kwargs: Other keyword arguments as described in the :func:`date` wrapper
        function, notably ``mutation_rate``, and ``population_size`` or ``priors``.
        Further arguments include ``time_units``, ``progress``, and
        ``record_provenance``.  The additional arguments ``return_model`` and
        ``return_likelihood`` can be used to return additional information (see below).
    :return:
        - **ts** (:class:`~tskit.TreeSequence`) -- a copy of the input tree sequence with
          updated node times based on the posterior mean, corrected where necessary to
          ensure that parents are strictly older than all their children by an amount
          given by the ``eps`` parameter.
        - **marginal_likelihood** (:py:class:`float`) -- (Only returned if
          ``return_likelihood`` is ``True``) The marginal likelihood of
          the mutation data given the inferred node times.
    """
    if Ne is not None:
        if population_size is not None:
            raise ValueError("Only provide one of Ne (deprecated) or population_size")
        else:
            population_size = Ne
    if eps is None:
        eps = DEFAULT_EPSILON
    if probability_space is None:
        probability_space = LOG_GRID

    dating_method = MaximizationMethod(
        tree_sequence,
        mutation_rate=mutation_rate,
        population_size=population_size,
        priors=priors,
        **kwargs,
    )
    result = dating_method.run(
        eps=eps,
        num_threads=num_threads,
        cache_inside=cache_inside,
        probability_space=probability_space,
    )
    return dating_method.parse_result(result, eps)


def inside_outside(
    tree_sequence,
    *,
    mutation_rate,
    population_size=None,
    priors=None,
    eps=None,
    num_threads=None,
    outside_standardize=None,
    ignore_oldest_root=None,
    probability_space=None,
    # below deliberately undocumented
    cache_inside=False,
    # Deprecated params
    Ne=None,
    # Other params documented in `.date()`
    **kwargs,
):
    """
    inside_outside(tree_sequence, *, mutation_rate, population_size=None, priors=None,\
        eps=None, num_threads=None, outside_standardize=None, ignore_oldest_root=None,\
        probability_space=None, **kwargs)

    Infer dates for nodes in a genealogical graph using the "inside outside" algorithm.
    This approximates the marginal posterior distribution of a node's age using an
    atomic discretization of time (e.g. point masses at particular timepoints).

    Currently, this estimation method comprises a single "inside" followed by a similar
    "outside" step. The inside step passes backwards in time from the samples to the
    roots of the graph,taking account of the distributions of times of each node's child
    (and if a ``mutation_rate`` is given, the the number of mutations on each edge).
    The outside step passes forwards in time from the roots, incorporating the time
    distributions for each node's parents. If there are (undirected) cycles in the
    underlying graph, this method does not provide a theoretically exact estimate
    of the marginal posterior distribution of node ages, but in practice it
    results in an accurate approximation.

    For example:

    .. code-block:: python

      new_ts = tsdate.inside_outside(ts, mutation_rate=1e-8, population_size=1e4)

    .. note::
        The prior parameters for each node-to-be-dated take the form of probabilities
        for each node at a set of discrete timepoints. If the ``priors`` parameter is
        used, it must specify an object constructed using :func:`build_prior_grid`
        (this can be used to define the number and position of the timepoints).
        If ``priors`` is not used, ``population_size`` must be provided,
        which is used to create a default prior derived from the conditional coalescent
        (tilted according to population size and weighted by the genomic
        span over which a node has a given number of descendant samples). This default
        prior assumes the nodes to be dated are all the non-sample nodes in the input
        tree sequence, and that they are contemporaneous.

    :param ~tskit.TreeSequence tree_sequence: The input tree sequence to be dated.
    :param float mutation_rate: The estimated mutation rate per unit of genome per
        unit time. If provided, the dating algorithm will use a mutation rate clock to
        help estimate node dates. Default: ``None``
    :param float or ~demography.PopulationSizeHistory population_size: The estimated
        (diploid) effective population size used to construct the (default) conditional
        coalescent prior. For a population with constant size, this can be given as a
        single value (for example, as commonly estimated by the observed genetic
        diversity of the sample divided by four-times the expected mutation rate).
        Alternatively, for a population with time-varying size, this can be given
        directly as a :class:`~demography.PopulationSizeHistory` object or a parameter
        dictionary passed to initialise a :class:`~demography.PopulationSizeHistory`
        object. The ``population_size`` parameter is only used when ``priors`` is
        ``None``. Conversely, if ``priors`` is not ``None``, no ``population_size``
        value should be specified.
    :param tsdate.node_time_class.NodeTimeValues priors: NodeTimeValues object containing
        the prior parameters for each node-to-be-dated. Note that different estimation
        methods may require different types of prior, as described in the documentation
        for each estimation method.
    :param float eps: The error factor in time difference calculations, and the
        minimum distance separating parent and child ages in the returned tree sequence.
        Default: None, treated as 1e-6.
    :param int num_threads: The number of threads to use when precalculating likelihoods.
        A simpler unthreaded algorithm is used unless this is >= 1. Default: None
    :param bool outside_standardize: Should the likelihoods be standardized during the
        outside step? This can help to avoid numerical under/overflow. Using
        unstandardized values is mostly useful for testing (e.g. to obtain, in the
        outside step, the total functional value for each node).
        Default: None, treated as True.
    :param bool ignore_oldest_root: Should the oldest root in the tree sequence be
        ignored in the outside algorithm (if ``"inside_outside"`` is used as the method).
        Ignoring outside root can provide greater stability when dating tree sequences
        inferred from real data, in particular if all local trees are assumed to coalesce
        in a single "grand MRCA", as in older versions of ``tsinfer``.
        Default: None, treated as False.
    :param string probability_space: Should the internal algorithm save
        probabilities in "logarithmic" (slower, less liable to to overflow) or
        "linear" space (fast, may overflow). Default: "logarithmic"
    :param \\**kwargs: Other keyword arguments as described in the :func:`date` wrapper
        function, notably ``mutation_rate``, and ``population_size`` or ``priors``.
        Further arguments include ``time_units``, ``progress``, and
        ``record_provenance``. The additional arguments ``return_model`` and
        ``return_likelihood`` can be used to return additional information (see below).
    :return:
        - **ts** (:class:`~tskit.TreeSequence`) -- a copy of the input tree sequence with
          updated node times based on the posterior mean, corrected where necessary to
          ensure that parents are strictly older than all their children by an amount
          given by the ``eps`` parameter.
        - **model** (:class:`~discrete.InOutModel`) -- (Only returned if ``return_model``
          is ``True``) The underlying object used to run the dating inference. This can
          then be queried e.g. for :meth:`~discrete.InOutModel.node_posteriors()`
        - **marginal_likelihood** (:py:class:`float`) -- (Only returned if
          ``return_likelihood`` is ``True``) The marginal likelihood of
          the mutation data given the inferred node times.
    """
    if Ne is not None:
        if population_size is not None:
            raise ValueError("Only provide one of Ne (deprecated) or population_size")
        else:
            population_size = Ne
    if eps is None:
        eps = DEFAULT_EPSILON
    if probability_space is None:
        probability_space = LOG_GRID
    if outside_standardize is None:
        outside_standardize = True
    if ignore_oldest_root is None:
        ignore_oldest_root = False
    dating_method = InsideOutsideMethod(
        tree_sequence,
        mutation_rate=mutation_rate,
        population_size=population_size,
        priors=priors,
        **kwargs,
    )
    result = dating_method.run(
        eps=eps,
        num_threads=num_threads,
        outside_standardize=outside_standardize,
        ignore_oldest_root=ignore_oldest_root,
        cache_inside=cache_inside,
        probability_space=probability_space,
    )
    return dating_method.parse_result(result, eps)


def variational_gamma(
    tree_sequence,
    *,
    mutation_rate,
    eps=None,
    max_iterations=None,
    rescaling_intervals=None,
    rescaling_iterations=None,
    match_segregating_sites=None,
    # deliberately undocumented parameters below. We may eventually document these
    max_shape=None,
    regularise_roots=None,
    singletons_phased=None,
    **kwargs,
):
    """
    variational_gamma(tree_sequence, *, mutation_rate, eps=None, max_iterations=None,\
            rescaling_intervals=None, **kwargs)

    Infer dates for nodes in a tree sequence using expectation propagation,
    which approximates the marginal posterior distribution of a given node's
    age with a gamma distribution. Convergence to the correct posterior moments
    is obtained by updating the distributions for node dates using several rounds
    of iteration. For example:

    .. code-block:: python

      new_ts = tsdate.variational_gamma(ts, mutation_rate=1e-8, max_iterations=10)

    A piecewise-constant uniform distribution is used as a prior for each
    node, that is updated via expectation maximization in each iteration.
    Node-specific priors are not currently supported.

    :param ~tskit.TreeSequence tree_sequence: The input tree sequence to be dated.
    :param float mutation_rate: The estimated mutation rate per unit of genome per
        unit time.
    :param float eps: The minimum distance separating parent and child ages in
        the returned tree sequence. Default: None, treated as 1e-6
    :param int max_iterations: The number of iterations used in the expectation
        propagation algorithm. Default: None, treated as 10.
    :param float rescaling_intervals: For time rescaling, the number of time
        intervals within which to estimate a rescaling parameter. Setting this to zero
        means that rescaling is not performed. Default ``None``, treated as 1000.
    :param float rescaling_iterations: The number of iterations for time rescaling.
        Setting this to zero means that rescaling is not performed. Default
        ``None``, treated as 5.
    :param bool match_segregating_sites: If ``True``, then time is rescaled
        such that branch- and site-mode segregating sites are approximately equal.
        If ``False``, time is rescaled such that branch- and site-mode root-to-leaf
        length are approximately equal, which gives unbiased estimates when there
        are polytomies. Default ``False``.
    :param \\**kwargs: Other keyword arguments as described in the :func:`date` wrapper
        function, including ``time_units``, ``progress``, and ``record_provenance``.
        The arguments ``return_model`` and ``return_likelihood`` can be
        used to return additional information (see below).
    :return:
        - **ts** (:class:`~tskit.TreeSequence`) -- a copy of the input tree sequence with
          updated node times based on the posterior mean, corrected where necessary to
          ensure that parents are strictly older than all their children by an amount
          given by the ``eps`` parameter.
        - **model** (:class:`~variational.ExpectationPropagationModel`) -- (Only returned
          if ``return_model`` is ``True``). The underlying object used to run the dating
          inference. This can then be queried e.g. for
          :meth:`~variational.ExpectationPropagationModel.node_posteriors()`
        - **marginal_likelihood** (:py:class:`float`) -- (Only returned if
          ``return_likelihood`` is ``True``) The marginal likelihood of
          the mutation data given the inferred node times. Not currently
          implemented for this method (set to ``None``)
    """
    if eps is None:
        eps = DEFAULT_EPSILON
    if max_iterations is None:
        max_iterations = DEFAULT_MAX_ITERATIONS
    if max_shape is None:
        # The maximum value for the shape parameter in the variational posteriors.
        # Equivalent to the maximum precision (inverse variance) on a logarithmic scale.
        max_shape = 1000
    if rescaling_intervals is None:
        rescaling_intervals = DEFAULT_RESCALING_INTERVALS
    if rescaling_iterations is None:
        rescaling_iterations = DEFAULT_RESCALING_ITERATIONS
    if match_segregating_sites is None:
        match_segregating_sites = False
    if regularise_roots is None:
        regularise_roots = True
    if singletons_phased is None:
        singletons_phased = True
    if tree_sequence.num_mutations == 0:
        raise ValueError(
            "No mutations present: these are required for the variational_gamma method"
        )
    dating_method = VariationalGammaMethod(
        tree_sequence, mutation_rate=mutation_rate, **kwargs
    )
    result = dating_method.run(
        eps=eps,
        max_iterations=max_iterations,
        max_shape=max_shape,
        rescaling_intervals=rescaling_intervals,
        rescaling_iterations=rescaling_iterations,
        match_segregating_sites=match_segregating_sites,
        regularise_roots=regularise_roots,
        singletons_phased=singletons_phased,
    )
    return dating_method.parse_result(result, eps)


estimation_methods = {
    "variational_gamma": variational_gamma,
    "inside_outside": inside_outside,
    "maximization": maximization,
}
"""
The names of available estimation methods, each mapped to a function to carry
out the appropriate method. Names can be passed as strings to the
:func:`~tsdate.date` function, or each named function can be called directly:

* :func:`tsdate.variational_gamma`: variational approximation, empirically most accurate.
* :func:`tsdate.inside_outside`: empirically better, theoretically problematic.
* :func:`tsdate.maximization`: worse empirically, especially with gamma approximated
  priors, but theoretically robust
"""


def date(
    tree_sequence,
    *,
    mutation_rate,
    recombination_rate=None,
    time_units=None,
    method=None,
    constr_iterations=None,
    return_model=None,
    return_likelihood=None,
    progress=None,
    record_provenance=True,
    # Other kwargs documented in the functions for each specific estimation-method
    **kwargs,
):
    """
    Infer dates for nodes in a genealogical graph (or :ref:`ARG<tutorials:sec_args>`)
    stored in the :ref:`succinct tree sequence<tskit:sec_introduction>` format.
    New times are assigned to nodes using the estimation algorithm specified by
    ``method`` (see note below). A ``mutation_rate`` must be given (the recombination_rate
    parameter, implementing a recombination clock, is unsupported at this
    time). Times associated with mutations and times associated
    with non-fixed (non-sample) nodes are overwritten. For example:

    .. code-block:: python

      mu = 1e-8
      new_ts = tsdate.date(ts, mutation_rate=mu)

    .. note::
        This is a wrapper for the named functions that are listed in
        :data:`~tsdate.core.estimation_methods`. Details and specific parameters for
        each estimation method are given in the documentation for those functions.

    :param ~tskit.TreeSequence tree_sequence: The input tree sequence to be dated (for
        example one with :data:`uncalibrated<tskit.TIME_UNITS_UNCALIBRATED>` node times).
    :param float mutation_rate: The estimated mutation rate per unit of genome per
        unit time (see individual methods)
    :param float recombination_rate: The estimated recombination rate per unit of genome
        per unit time. If provided, the dating algorithm will use a recombination rate
        clock to help estimate node dates. Default: ``None`` (not currently implemented)
    :param str time_units: The time units used by the ``mutation_rate`` and
        ``recombination_rate`` values, and stored in the ``time_units`` attribute of the
        output tree sequence. If the conditional coalescent prior is used,
        then this is also applies to the value of ``population_size``, which in
        standard coalescent theory is measured in generations. Therefore if you
        wish to use mutation and recombination rates measured in (say) years,
        and are using the conditional coalescent prior, the ``population_size``
        value which you provide must be scaled by multiplying by the number of
        years per generation. If ``None`` (default), assume ``"generations"``.
    :param string method: What estimation method to use. See
        :data:`~tsdate.core.estimation_methods` for possible values.
        If ``None`` (default) the "variational_gamma" method is currently chosen.
    :param bool return_model: If ``True``, instead of returning just a dated tree
        sequence, return a tuple of ``(dated_ts, model)``.
        Default: None, treated as False.
    :param int constr_iterations: The maximum number of constrained least
        squares iterations to use prior to forcing positive branch lengths.
        Default: None, treated as 0.
    :param bool return_likelihood: If ``True``, return the log marginal likelihood
        from the inside algorithm in addition to the dated tree sequence. If
        ``return_model`` is also ``True``, then the marginal likelihood
        will be the last element of the tuple. Default: None, treated as False.
    :param bool progress: Show a progress bar. Default: None, treated as False.
    :param bool record_provenance: Should the tsdate command be appended to the
        provenence information in the returned tree sequence?
        Default: None, treated as True.
    :param \\**kwargs: Other keyword arguments specific to the
        :data:`estimation method<tsdate.core.estimation_methods>` used. These are
        documented in those specific functions.
    :return:
        A copy of the input tree sequence but with updated node times, or (if
        ``return_model`` or ``return_likelihood`` is True) a tuple of that
        tree sequence plus a model object and/or the
        marginal likelihood given the mutations on the tree sequence.
    """
    # Only the .date() wrapper needs to consider the deprecated "Ne" param
    if method is None:
        method = "variational_gamma"
    if method not in estimation_methods:
        raise ValueError(f"method must be one of {list(estimation_methods.keys())}")

    return estimation_methods[method](
        tree_sequence,
        mutation_rate=mutation_rate,
        recombination_rate=recombination_rate,
        time_units=time_units,
        progress=progress,
        constr_iterations=constr_iterations,
        return_model=return_model,
        return_likelihood=return_likelihood,
        record_provenance=record_provenance,
        **kwargs,
    )
