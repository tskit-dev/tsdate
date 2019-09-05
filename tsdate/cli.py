# MIT License
#
# Copyright (c) 2019 Anthony Wilder Wohns
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
import sys

import tskit

import tsdate


def exit(message):
    """
    Exit with the specified error message, setting error status.
    """
    sys.exit("{}: {}".format(sys.argv[0], message))


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
    parser.add_argument('-g', '--time-grid', type=str, default='adaptive',
                        help="specify a uniform time grid")
    parser.add_argument('-s', '--slices', type=int, default=50,
                        help="intervals in time grid")
    parser.add_argument('-e', '--epsilon', type=float, default=1e-6,
                        help="value to add to dt")
    parser.add_argument('-t', '--num-threads', type=int, default=0,
                        help="number of threads to use")
    parser.add_argument('-p', '--progress', action='store_true',
                        help="show progress bar")
    return parser


def run_age_inference(args):
    try:
        ts = tskit.load(args.ts)
    except tskit.FileFormatError as ffe:
        exit("Error loading '{}: {}".format(args.ts, ffe))
    dated_ts = tsdate.age_inference(
        ts, args.Ne, args.mutation_rate, args.recombination_rate,
        args.time_grid, args.slices, args.epsilon, args.num_threads,
        args.progress)
    dated_ts.dump(args.output)


def main(args):
    # Load tree sequence
    run_age_inference(args)


def tsdate_main(arg_list=None):
    parser = tsdate_cli_parser()
    args = parser.parse_args(arg_list)
    main(args)
