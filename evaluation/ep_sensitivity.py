"""
How much do whole EP sweeps amplify last-bit differences?

The mutational target of every edge (the input of the likelihood pass) is perturbed
by one ulp, and the trajectory of node means is compared with the unperturbed run,
using tsdate's production `ExpectationPropagation`. This is the noise level that a
reimplementation with differently rounded intermediates has to live with, and the
reason that multi-sweep comparisons in `tests/test_oracle_variational.py` use
relative tolerances (1e-4 after the first sweep, 1e-6 later) and not ulps.

Prints, for each tree sequence, the maximum relative change in node means after
selected sweeps, and whether the pattern of skipped updates (NaN edge log-normalising
constants) changes.

Example:
  python evaluation/ep_sensitivity.py --quick
  python evaluation/ep_sensitivity.py --sizes 20 100 --seeds 1 2 3 --sweeps 1 5 10
"""

import argparse

import common
import numpy as np

from tsdate.variational import ExpectationPropagation


def node_means(posterior):
    """Means of the gamma distributions with natural parameters `posterior`"""
    with np.errstate(all="ignore"):
        return (posterior[:, 0] + 1) / posterior[:, 1]


def run_pair(ts, mutation_rate, num_sweeps, max_shape, min_step):
    """
    Run EP with and without the perturbation; yield, after each sweep, the maximum
    relative difference in node means and whether the skipped updates differ.
    """
    eps = []
    for perturb in (False, True):
        ep = ExpectationPropagation(ts, mutation_rate=mutation_rate)
        if perturb:
            target = ep.edge_likelihoods[:, 1]
            ep.edge_likelihoods[:, 1] = np.nextafter(target, np.inf)
        eps.append(ep)
    for _ in range(num_sweeps):
        for ep in eps:
            ep.iterate(max_shape=max_shape, min_step=min_step)
        base, perturbed = (node_means(ep.node_posterior) for ep in eps)
        ok = np.isfinite(base) & np.isfinite(perturbed)
        change = np.abs(perturbed[ok] - base[ok]) / np.abs(base[ok])
        skipped = [np.isnan(ep.edge_logconst) for ep in eps]
        yield float(change.max(initial=0.0)), not np.array_equal(*skipped)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[20, 100])
    parser.add_argument("--length", type=float, default=2e5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--sweeps", type=int, nargs="+", default=[1, 2, 5, 10])
    parser.add_argument("--mutation-rate", type=float, default=1e-8)
    parser.add_argument("--max-shape", type=float, default=1000.0)
    parser.add_argument("--min-step", type=float, default=0.1)
    parser.add_argument("--quick", action="store_true", help="one small simulation")
    args = parser.parse_args()
    if args.quick:
        args.sizes, args.seeds, args.sweeps, args.length = [10], [1], [1, 3], 1e5
    header = "".join(f"{f'sweep {s}':>12}" for s in args.sweeps)
    print(f"{'samples':>8}{'seed':>6}{header}  skipped-pattern-changed")
    for n in args.sizes:
        for seed in args.seeds:
            ts = common.simulate(n, args.length, args.mutation_rate, seed)
            runs = run_pair(
                ts, args.mutation_rate, max(args.sweeps), args.max_shape, args.min_step
            )
            results = list(runs)
            row = "".join(f"{results[s - 1][0]:12.2e}" for s in args.sweeps)
            changed = any(r[1] for r in results)
            print(f"{n:>8}{seed:>6}{row}  {changed}")


if __name__ == "__main__":
    main()
