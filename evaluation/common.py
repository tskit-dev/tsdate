"""Shared helpers for the evaluation scripts."""

import datetime
import hashlib
import os
import pathlib
import subprocess

import msprime
import tskit

HERE = pathlib.Path(__file__).parent
CACHE = HERE / "cache"
RESULTS = HERE / "results"
NE = 10_000
RECOMB_RATE = 1e-8


def provenance():
    """Columns identifying the code version, to allow comparing runs over commits."""
    import tsdate

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=HERE, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no", str(HERE.parent)],
            cwd=HERE,
            text=True,
        ).strip()
        sha += "-dirty" if dirty else ""
    except Exception:
        sha = ""
    return {
        "tsdate_version": tsdate.__version__,
        "git_sha": sha,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }


# The coalescent approximation breaks down in the most recent generations for large
# samples (many near-simultaneous coalescences), so simulate those with a
# discrete-time Wright-Fisher model.
DTWF_GENERATIONS = 20


def simulate(n, length, mutation_rate=1e-8, seed=1):
    """Simulate (and cache) a tree sequence with n diploid individuals."""
    CACHE.mkdir(exist_ok=True)
    key = f"sim_dtwf{DTWF_GENERATIONS}_{n}_{length:g}_{mutation_rate:g}_{seed}"
    path = CACHE / (hashlib.md5(key.encode()).hexdigest()[:10] + f"_{key}.trees")
    if path.exists():
        return tskit.load(path)
    ts = msprime.sim_ancestry(
        n,
        population_size=NE,
        recombination_rate=RECOMB_RATE,
        sequence_length=length,
        model=[
            msprime.DiscreteTimeWrightFisher(duration=DTWF_GENERATIONS),
            msprime.StandardCoalescent(),
        ],
        random_seed=seed,
    )
    ts = msprime.sim_mutations(
        ts, rate=mutation_rate, model=msprime.BinaryMutationModel(), random_seed=seed
    )
    # Write atomically so that parallel workers never read a partial file
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    ts.dump(tmp)
    os.replace(tmp, path)
    return ts


def infer_topology(ts, num_threads=1):
    """Infer a topology from the variation in ts using tsinfer."""
    import tsinfer

    sd = tsinfer.SampleData.from_tree_sequence(ts, use_sites_time=False)
    return tsinfer.infer(sd, num_threads=num_threads).simplify(keep_unary=False)
