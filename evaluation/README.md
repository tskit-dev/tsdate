# Evaluation

Scripts to benchmark and validate tsdate **by hand**. They are not part of the
package or of the unit tests (pytest only collects `tests/`), and need extra
dependencies: `pip install -r evaluation/requirements.txt`.

Run from the repository root; results go to `evaluation/results/` (git-ignored, except `performance*.csv`/`.png`, which are committed to track progress),
and simulations are cached in `evaluation/cache/`. Simulations use a discrete-time
Wright-Fisher model for the most recent 20 generations (so node times are integers
there, and the coalescent approximation does not give sub-generation nodes), then the
standard coalescent.

| Script | Purpose |
|---|---|
| `accuracy.py` | Sweep a parameter (`sample_size`, `length`, `mutation_rate`, `max_iterations`) for `variational_gamma`, comparing estimated to true node times. `--modes true` dates the true topology; `inferred` dates a tsinfer topology and compares to best-matching true nodes via [tscompare](https://github.com/tskit-dev/tscompare) (see issue #301). Writes one row per node (`accuracy_*.csv`). It also compares mutation times (`mutations_*.csv`): sites with a single mutation are matched by position, and tsdate's posterior mean mutation time is compared with the midpoint of the true edge the mutation sits on (the best any method can do; the exact simulated time is also saved as `true_time_exact`). `plot.py` scores scatter, RMSE and bias against the midpoint but PIT and interval coverage against the exact time, since the posterior is for the realised mutation time; `--mutation-truth midpoint` or `exact` forces one truth everywhere. Mutation comparison needs no node matching in either mode. |
| `afs.py` | Compare the observed site allele frequency spectrum with the expected one from branch lengths (branch AFS x mutation rate) in genomic windows, for the true times, the true topology dated by tsdate, and a tsinfer topology dated by tsdate. Writes `afs_*.csv`. |
| `performance.py` | Wall time and peak RSS, each case in a fresh process. |
| `compare.py` | Compares two `performance.py` CSVs case by case (see below). |
| `plot.py` | Reads `results/*.csv` and writes PNGs: error/bias/interval coverage vs parameter (`accuracy_*_summary.png`: top row node times, bottom row mutation times), estimated-vs-true scatter (`accuracy_*_scatter.png`: columns are swept values, rows are node/mutation times for true/tsinferred topology), PIT calibration histograms (`accuracy_*_calibration.png`: node row, mutation row), AFS observed-vs-expected (`afs_*.png`), time and memory scaling. |
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

#### Comparing against the stored results

`evaluation/results/performance.csv` and `performance.png` are committed, so they act as
a reference to check progress against. Each row records the machine (`machine`, `cpu`,
`cpu_cores`, `ram_gb`, `os`, `python`) as well as the tsdate version and git SHA, and
`compare.py` prints the machine of each file. To test the current code against the
stored run, write the new results to a *different* file (otherwise you overwrite the
stored reference) and compare:

```
python evaluation/performance.py --repeats 3 --output new.csv   # same options as stored run
python evaluation/compare.py evaluation/results/performance.csv evaluation/results/new.csv
```

The stored run's sweep (sizes, lengths, `--inferred`) is in the CSV, so use the same
options; only cases present in both files are compared. If the CPUs differ, `compare.py`
prints a note: the ratios then include hardware effects (a fanless laptop may also
throttle), so interpret them with care. To compare with an older commit instead, use
`git show <commit>:evaluation/results/performance.csv > old.csv`. `new.csv` is
git-ignored. To update the stored reference, re-run without `--output`, run `plot.py`,
and commit the two `performance.*` files. The plot shows mean ± standard error over the
repeats (use 5 or more for a reference run).

Add `--quick` to `accuracy.py` / `performance.py` for a fast smoke test. Quick runs write
to separate `*_quick.csv` files (so they never overwrite full results), and `plot.py`
labels their figures "quick run" and names them `*_quick*.png`.
