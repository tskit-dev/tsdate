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
# sys.path.insert(1, '../tsdate')
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
                        help="Tree sequence from which we estimate age")
    parser.add_argument('output',
                        help="path and name of output file")
    parser.add_argument('-n', '--Ne', type=float, default=10000,
                        help="effective population size")
    parser.add_argument('-m', '--mutation-rate', type=float, default=None,
                        help="mutation rate")
    parser.add_argument('-r', '--recombination-rate', type=float,
                        default=None, help="recombination rate")
    parser.add_argument('-e', '--epsilon', type=float, default=1e-6,
                        help="value to add to dt")
    parser.add_argument('-t', '--num-threads', type=int, default=1,
                        help="number of threads to use")
    parser.add_argument('--probability-space', type=str, default='logarithmic',
                        help="Should the internal algorithm save probabilities in \
                        'logarithmic' (slower, less liable to to overflow) or 'linear' \
                        space (faster, may overflow).")
    parser.add_argument('--method', type=str, default='inside_outside',
                        help="Specify which estimation method to use: can be \
                        'inside_outside' or 'maximization'.")
    parser.add_argument('-p', '--progress', action='store_true',
                        help="show progress bar")
    parser.add_argument('-v', '--verbosity', type=int, default=0,
                        help="how much verbosity to output")
    return parser


def run_date(args):
    setup_logging(args)
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


def main(args):
    # Load tree sequence
    run_date(args)


def tsdate_main(arg_list=None):
    parser = tsdate_cli_parser()
    args = parser.parse_args(arg_list)
    main(args)
