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

import tskit

import tsdate
from . import exceptions

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
    parser.add_argument('ts', type=str,
                        help="Tree sequence from which we estimate age")
    parser.add_argument('-c', '--clock', type=str, default='mutation',
                        help="mutation, recombination, or combination")
    parser.add_argument('-n', '--Ne', type=float, default=10000,
                        help="effective population size")
    parser.add_argument('-u', '--uniform', action='store_true',
                        help="specify a uniform time grid")
    parser.add_argument('-t', '--theta', type=float, default=0.0004,
                        help="population scaled mutation rate")
    parser.add_argument('-r', '--rho', type=float, default=0.0004,
                        help="population scaled recombination rate")
    parser.add_argument('-i', '--del_p', type=float, default=0.02,
                        help="intervals in time grid")
    parser.add_argument('-o', '--output', type=str, default="output",
                        help="name of output file")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-s', '--save_ts', action='store_true',
        help="Save a tree sequence with point estimates for node age")
    group.add_argument(
        '-d', '--node_dist', action='store_true',
        help="Output distributions of node ages")
    return parser


def run_age_inference(args):
    try:
        ts = tskit.load(args.ts)
    except exceptions.FileFormatError as ffe:
        exit("Error loading '{}: {}".format(args.ts, ffe))
    posterior, time_grid, mn_post = tsdate.age_inference(
        ts, args.uniform, args.clock, args.Ne, args.theta, args.rho,
        args.del_p, args.output)
    new_mn_post = tsdate.restrict_ages_topo(ts, mn_post, time_grid)
    dated_ts = tsdate.return_ts(ts, time_grid, new_mn_post, args.Ne)
    print(dated_ts)

def main(args):
    # Load tree sequence
    if args.output:
        run_age_inference(args)


def tsdate_main(arg_list=None):
    parser = tsdate_cli_parser()
    args = parser.parse_args(arg_list)
    main(args)
