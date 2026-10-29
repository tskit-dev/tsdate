# Evaluation

Scripts to benchmark and validate tsdate **by hand**. They are not part of the
package or of the unit tests (pytest only collects `tests/`), and need extra
dependencies: `pip install -r evaluation/requirements.txt`.

Run from the repository root; results go to `evaluation/results/` (git-ignored),
and simulations are cached in `evaluation/cache/`. Simulations use a discrete-time
Wright-Fisher model for the most recent 20 generations (so node times are integers
there, and the coalescent approximation does not give sub-generation nodes), then the
standard coalescent.

| Script | Purpose |
|---|---|
| `accuracy.py` | Sweep a parameter (`sample_size`, `length`, `mutation_rate`, `max_iterations`) for `variational_gamma`, comparing estimated to true node times. `--modes true` dates the true topology; `inferred` dates a tsinfer topology and compares to best-matching true nodes via [tscompare](https://github.com/tskit-dev/tscompare) (see issue #301). Writes one row per node (`accuracy_*.csv`). It also compares mutation times (`mutations_*.csv`): sites with a single mutation are matched by position, and tsdate's posterior mean mutation time is compared with the midpoint of the true edge the mutation sits on (the best any method can do; the exact simulated time is also saved as `true_time_exact`). `plot.py` scores scatter, RMSE and bias against the midpoint but PIT and interval coverage against the exact time, since the posterior is for the realised mutation time; `--mutation-truth midpoint` or `exact` forces one truth everywhere, so no node matching is needed in either mode. |
| `performance.py` | Wall time and memory increase during dating, each case in a fresh process (numba code is warmed up first and cached on disk, so compilation is excluded). |
| `compare.py` | Compares two `performance.py` CSVs case by case (see below). |
| `plot.py` | Reads `results/*.csv` and writes PNGs: error/bias/interval coverage vs parameter (`accuracy_*_summary.png`: top row node times, bottom row mutation times), estimated-vs-true scatter (`accuracy_*_scatter.png`: columns are swept values, rows are node/mutation times for true/tsinferred topology), PIT calibration histograms (`accuracy_*_calibration.png`: node row, mutation row), time and memory scaling. |
| `validation.py` | Stub: numerical checks too slow for the unit tests (issue #384). |

Each CSV row records the tsdate version, git SHA and timestamp.

```
python evaluation/accuracy.py --sweep sample_size --modes true inferred --seeds 3 --max-seeds 250 --cores 8
python evaluation/performance.py
python evaluation/plot.py
```
### Comparing performance across commits

`performance.py` records the git SHA in every row (suffixed `-dirty` if tracked files
have uncommitted changes), and `--output` names the CSV. To test whether a change
affects speed or memory, run the same sweep on each version and compare:

```
git checkout main
python evaluation/performance.py --repeats 3 --output base.csv
git checkout my-branch
python evaluation/performance.py --repeats 3 --output new.csv
python evaluation/compare.py evaluation/results/base.csv evaluation/results/new.csv
```

For each (length, sample size) it prints the median time and memory increase for both
versions and the ratio new/base (below 1 is better). Differences smaller than the
min-to-max spread of the repeats are shown as `~` (noise), so use at least 3 repeats and
keep the machine otherwise idle. Use identical options (`--sizes`, `--lengths`,
`--inferred`) for both runs; only cases present in both files are compared. Memory is
the RSS increase during dating, not total process memory.

Add `--quick` to `accuracy.py` / `performance.py` for a fast smoke test. Quick runs write
to separate `*_quick.csv` files (so they never overwrite full results), and `plot.py`
labels their figures "quick run" and names them `*_quick*.png`.
