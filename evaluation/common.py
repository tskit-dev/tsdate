"""Shared helpers for the evaluation scripts."""

import datetime
import hashlib
import os
import pathlib
import platform
import subprocess

import msprime
import tskit

HERE = pathlib.Path(__file__).parent
CACHE = HERE / "cache"
RESULTS = HERE / "results"
NE = 10_000
RECOMB_RATE = 1e-8


def cpu_name():
    try:
        if platform.system() == "Darwin":
            cmd = ["sysctl", "-n", "machdep.cpu.brand_string"]
            return subprocess.check_output(cmd, text=True).strip()
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def machine_model():
    """Hardware model, e.g. "MacBook Air (Mac14,2)"; useful as laptops may throttle."""
    try:
        if platform.system() == "Darwin":
            out = subprocess.check_output(
                ["system_profiler", "SPHardwareDataType"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            info = dict(
                (k.strip(), v.strip())
                for k, _, v in (line.partition(":") for line in out.splitlines())
            )
            return f"{info['Model Name']} ({info['Model Identifier']})"
        if platform.system() == "Linux":
            dmi = pathlib.Path("/sys/devices/virtual/dmi/id")
            parts = [
                (dmi / f).read_text().strip() for f in ("sys_vendor", "product_name")
            ]
            return " ".join(parts)
    except Exception:
        pass
    return ""


def total_ram_gb():
    try:
        if platform.system() == "Darwin":
            cmd = ["sysctl", "-n", "hw.memsize"]
            return round(int(subprocess.check_output(cmd, text=True)) / 2**30)
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30)
    except Exception:
        return None


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
        "machine": machine_model(),
        "cpu": cpu_name(),
        "cpu_cores": os.cpu_count(),
        "ram_gb": total_ram_gb(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
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
