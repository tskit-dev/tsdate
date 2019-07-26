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
Command line interface for tstime.
"""
import argparse

import tskit

import tstime


def tstime_cli_parser():
    parser = argparse.ArgumentParser(
        description="Set up base data, generate inferred datasets,\
                     and process datasets.")
    parser.add_argument('--ts', dest='ts', type=str,
                        help="path to the the tree sequence file")
    parser.add_argument('--clock', dest='clock', type=str,
                        help="mutation, recombination, or combination")
    parser.add_argument('--grid', dest='grid', type=str, default="union",
                        help="union or uniform based time grid")
    parser.add_argument('--theta', dest='theta', type=float, default=0.0004,
                        help="population scaled mutation rate")
    parser.add_argument('--rho', dest='rho', type=float, default=0.0004,
                        help="population scaled recombination rate")
    parser.add_argument('--del_p', dest='del_p', type=float, default=0.02,
                        help="intervals in time grid")
    parser.add_argument('--output', dest='output', type=str, default="output",
                        help="name of output csv")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--output_ts", action='store_true',
        help="Output a tree sequence with point estimates for node age")
    group.add_argument(
        "--node_dist", action='store_true',
        help="Output distributions of node ages")
    return parser


def main(args):
    # Load tree sequence
    try:
        ts = tskit.load(args['ts'])
    except KeyboardInterrupt:
        print("Cannot open tree sequence file")
    if args.output_ts:
        tstime.age_inference(
            ts, args['grid'], args['clock'], args['theta'], args['rho'],
            args['del_p'], args['output'])


def ts_time(arg_list=None):
    parser = tstime_cli_parser()
    args = parser.parse_args(arg_list)
    main(args)
