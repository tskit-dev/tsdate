# MIT License
#
# Copyright (c) 2024 Tskit Developers
# Copyright (c) 2020-2024 University of Oxford
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
Command line interface for tsdate.
"""
import argparse
import logging
import sys

import tskit

import tsdate
from . import core

logger = logging.getLogger(__name__)
log_format = "%(asctime)s %(levelname)s %(message)s"


def error_exit(message):
    """
    Exit with the specified error message, setting error status.
    """
    sys.exit(f"{sys.argv[0]}: {message}")


def setup_logging(args):
    log_level = "WARN"
    if args.verbosity > 0:
        log_level = "INFO"
    if args.verbosity > 1:
        log_level = "DEBUG"
    logging.basicConfig(level=log_level, format=log_format)
    return log_level


def tsdate_cli_parser():
    top_parser = argparse.ArgumentParser(
        description=(
            "This is the command line interface for tsdate, a tool to date "
            "tree sequences."
        ),
    )
    top_parser.add_argument(
        "-V", "--version", action="version", version=f"%(prog)s {tsdate.__version__}"
    )

    subparsers = top_parser.add_subparsers(dest="subcommand")
    subparsers.required = True

    parser = subparsers.add_parser(
        "date",
        help=(
            "Takes an inferred tree sequence topology and "
            "returns a dated tree sequence."
        ),
    )

    parser.add_argument(
        "tree_sequence",
        help=(
            "The path and name of the input tree sequence for which "
            "node ages are estimated."
        ),
    )
    parser.add_argument(
        "output",
        help=(
            "The path and name of output file where the dated tree "
            "sequence will saved."
        ),
    )
    parser.add_argument(
        "deprecated_population_size",
        type=float,
        nargs="?",
        help="Deprecated positional argument, left for backwards compatibility.",
    )
    parser.add_argument(
        "-m",
        "--mutation-rate",
        type=float,
        default=None,
        help=(
            "The estimated mutation rate per unit of genome per "
            "generation. If provided, the dating algorithm will use a "
            "mutation rate clock to help estimate node dates. Default: None"
        ),
    )
    parser.add_argument(
        "-r",
        "--recombination-rate",
        type=float,
        default=None,
        help=(
            "The estimated recombination rate per unit "
            "of genome per generation. If provided, the dating algorithm "
            "will use a recombination rate clock to help estimate node "
            "dates. Default: None"
        ),
    )
    parser.add_argument(
        "-e",
        "--epsilon",
        type=float,
        default=core.DEFAULT_EPSILON,
        help=(
            "Specify minimum distance separating time points. Also "
            "specifies the error factor in time difference calculations. "
            f"Default: {core.DEFAULT_EPSILON}"
        ),
    )
    parser.add_argument(
        "--method",
        choices=["inside_outside", "maximization", "variational_gamma"],
        default="variational_gamma",
        help=(
            "Specify which estimation method to use: "
            "'variational_gamma' is a fast continuous-time approximation' "
            "'inside_outside' is a discrete-time version but theoretically problematic; "
            "'maximization' is worse empirically, especially with a gamma prior, but "
            "theoretically robust; Current default: 'variational_gamma'"
        ),
    )
    parser.add_argument(
        "-p", "--progress", action="store_true", help="Show progress bar."
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        action="count",
        default=0,
        help="How much verbosity to output.",
    )
    parser.add_argument(
        "--rescaling-intervals",
        type=float,
        help=(
            "The number of time intervals within which to estimate a time scaling"
            f"parameter. Default: None treated as {core.DEFAULT_RESCALING_INTERVALS}"
        ),
        default=None,
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        help=(
            "The number of iterations used in the expectation propagation "
            f"algorithm. Default: None treated as {core.DEFAULT_MAX_ITERATIONS}"
        ),
        default=None,
    )
    # TODO array specification from file?
    parser.add_argument(
        "-n",
        "--population_size",
        type=float,
        default=None,
        help=(
            "Estimated effective (diploid) population size. Ignored for the "
            "'variational_gamma' method, but required otherwise. Default: None"
        ),
    )
    parser.add_argument(
        "-t",
        "--num-threads",
        type=int,
        default=None,
        help=(
            "The number of threads to use. A simpler unthreaded algorithm is used "
            "unless this is >= 1. Not relevant for the 'variational_gamma' method. "
            "Default: None"
        ),
    )
    parser.add_argument(
        "--probability-space",
        type=str,
        default=None,
        help=(
            "Should the internal algorithm save probabilities in "
            "'logarithmic' (slower, less liable to to overflow) or 'linear' "
            "space (faster, may overflow). Not relevant for the "
            "'variational_gamma' method. Default: None treated as 'logarithmic'"
        ),
    )
    parser.set_defaults(runner=run_date)

    parser = subparsers.add_parser(
        "preprocess", help=("Remove regions without data from an input tree sequence.")
    )
    parser.add_argument("tree_sequence", help="The tree sequence to preprocess.")
    parser.add_argument(
        "output",
        help=(
            "The path and name of output file where the preprocessed "
            "tree sequence will saved."
        ),
    )
    parser.add_argument(
        "--minimum_gap",
        type=float,
        help=(
            "The minimum gap between sites to trim from the tree "
            "sequence. Default: 1000000"
        ),
        default=1000000,
    )
    parser.add_argument(
        "--trim-telomeres",
        type=bool,
        help=(
            "Should all material before the first site and after the "
            "last site be trimmed, regardless of the length of these "
            "regions. Default: True"
        ),
        default=True,
    )
    parser.add_argument(
        "--split-disjoint",
        type=bool,
        help=(
            "Should disjoint nodes, that disappear from the trees then "
            "reappear further along the genome, be split into separate nodes. "
            "Default: True"
        ),
        default=True,
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        action="count",
        default=0,
        help="How much verbosity to output (max is -vv).",
    )
    parser.set_defaults(runner=run_preprocess)
    return top_parser


def run_date(args):
    if args.deprecated_population_size is not None:
        error_exit(
            "Specifying the population size without prefixing by `-n` is "
            f"deprecated. Please use `-n {args.deprecated_population_size}` instead."
        )
    try:
        ts = tskit.load(args.tree_sequence)
    except tskit.FileFormatError as ffe:
        error_exit(f"FileFormatError loading '{args.tree_sequence}: {ffe}")
    if args.method == "variational_gamma":
        # TODO - warn about other non-relevant options
        if args.population_size is not None:
            error_exit(
                "The population_size is not currently required for 'variational_gamma'"
            )
        if args.num_threads is not None:
            error_exit(
                "Multiple threads cannot be used in the 'variational_gamma' method"
            )
        if args.probability_space is not None:
            error_exit(
                "The probability_spaces parameter is irrelevant for 'variational_gamma'"
            )
        params = dict(
            recombination_rate=args.recombination_rate,
            method=args.method,
            eps=args.epsilon,
            progress=args.progress,
            max_iterations=args.max_iterations,
            rescaling_intervals=args.rescaling_intervals,
        )
    else:
        if args.rescaling_intervals is not None:
            error_exit(
                "rescaling_intervals is not currently used in discrete-time methods"
            )
            print("FOOOO")
        if args.max_iterations is not None:
            error_exit("max_iterations is not currently used in discrete-time methods")
        params = dict(
            population_size=args.population_size,
            recombination_rate=args.recombination_rate,
            method=args.method,
            eps=args.epsilon,
            progress=args.progress,
            probability_space=args.probability_space,
            num_threads=args.num_threads,
        )
    dated_ts = tsdate.date(ts, mutation_rate=args.mutation_rate, **params)
    dated_ts.dump(args.output)


def run_preprocess(args):
    try:
        ts = tskit.load(args.tree_sequence)
    except tskit.FileFormatError as ffe:
        error_exit(f"FileFormatError loading '{args.tree_sequence}: {ffe}")
    snipped_ts = tsdate.preprocess_ts(
        ts, minimum_gap=args.minimum_gap, remove_telomeres=args.trim_telomeres
    )
    snipped_ts.dump(args.output)


def tsdate_main(arg_list=None):
    parser = tsdate_cli_parser()
    args = parser.parse_args(arg_list)
    setup_logging(args)
    args.runner(args)
