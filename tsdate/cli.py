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
Command line interface for tsdate.
"""
import argparse
import logging
import sys

import tskit
import tsdate

logger = logging.getLogger(__name__)
log_format = '%(asctime)s %(levelname)s %(message)s'


def exit(message):
    """
    Exit with the specified error message, setting error status.
    """
    sys.exit("{}: {}".format(sys.argv[0], message))


def setup_logging(args):
    log_level = "WARN"
    if args.verbosity > 0:
        log_level = "INFO"
    if args.verbosity > 1:
        log_level = "DEBUG"
    logging.basicConfig(level=log_level, format=log_format)


def tsdate_cli_parser():
    parser = argparse.ArgumentParser(
        description="Set up base data, generate inferred datasets,\
                     and process datasets.")
    parser.add_argument(
        "-V", "--version", action='version',
        version='%(prog)s {}'.format(tsdate.__version__))
    parser.add_argument('ts',
                        help="The path and name of the input tree sequence from which \
                        we estimate node ages.")
    parser.add_argument('output',
                        help="The path and name of output file where the dated tree \
                        sequence will saved.")
    parser.add_argument('Ne', type=float,
                        help="estimated effective (diploid) population size.")
    parser.add_argument('-m', '--mutation-rate', type=float, default=None,
                        help="The estimated mutation rate per unit of genome per \
                        generation. If provided, the dating algorithm will use a \
                        mutation rate clock to help estimate node dates.")
    parser.add_argument('-r', '--recombination-rate', type=float,
                        default=None, help="The estimated recombination rate per unit \
                        of genome per generation. If provided, the dating algorithm \
                        will  use a recombination rate clock to help estimate node \
                        dates.")
    parser.add_argument('-e', '--epsilon', type=float, default=1e-6,
                        help="Specify minimum distance separating time points. Also \
                        specifies the error factor in time difference calculations.")
    parser.add_argument('-t', '--num-threads', type=int, default=None,
                        help="The number of threads to use. A simpler unthreaded \
                        algorithm is used unless this is >= 1 (default: None).")
    parser.add_argument('--probability-space', type=str, default='logarithmic',
                        help="Should the internal algorithm save probabilities in \
                        'logarithmic' (slower, less liable to to overflow) or 'linear' \
                        space (faster, may overflow).")
    parser.add_argument('--method', type=str, default='inside_outside',
                        help="Specify which estimation method to use: can be \
                        'inside_outside' (empirically better, theoretically \
                        problematic) or 'maximization' (worse empirically, especially \
                        with a gamma approximated prior, but theoretically robust). \
                        Default: 'inside_outside.'")
    parser.add_argument('-p', '--progress', action='store_true',
                        help="Show progress bar.")
    parser.add_argument('-v', '--verbosity', type=int, default=0,
                        help="How much verbosity to output.")
    return parser


def run_date(args):
    try:
        ts = tskit.load(args.ts)
    except tskit.FileFormatError as ffe:
        exit("Error loading '{}: {}".format(args.ts, ffe))
    dated_ts = tsdate.date(
        ts, args.Ne, mutation_rate=args.mutation_rate,
        recombination_rate=args.recombination_rate,
        probability_space=args.probability_space, method=args.method,
        eps=args.epsilon, num_threads=args.num_threads,
        progress=args.progress)
    dated_ts.dump(args.output)


def tsdate_main(arg_list=None):
    parser = tsdate_cli_parser()
    args = parser.parse_args(arg_list)
    setup_logging(args)
    run_date(args)
