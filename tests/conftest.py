import msprime
import numpy as np
import pytest
from oracle.implementations import IMPLEMENTATIONS


def pytest_addoption(parser):
    parser.addoption(
        "--make-files",
        action="store_true",
        default=False,
        help="generate static tree sequences used for testing",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--make-files"):
        # --make-files given in cli: only run tests marked @pytest.mark.makefiles
        skip_normal_tests = pytest.mark.skip(
            reason="--make-files specified, so other tests skipped"
        )
        for item in items:
            if "makefiles" not in item.keywords:
                item.add_marker(skip_normal_tests)
    else:
        skip_make_files = pytest.mark.skip(
            reason="specify --make-files to (re)create various files used for testing"
        )
        for item in items:
            if "makefiles" in item.keywords:
                item.add_marker(skip_make_files)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "makefiles: mark test to run only when --make-files option given"
    )


@pytest.fixture(params=list(IMPLEMENTATIONS), ids=list(IMPLEMENTATIONS))
def impl(request):
    """Namespace of an implementation (see oracle/implementations.py)."""
    return IMPLEMENTATIONS[request.param]


@pytest.fixture
def rng():
    return np.random.default_rng(1024)


def _small_ts(samples, seed, mutation_rate=2e-8):
    ts = msprime.sim_ancestry(
        samples=samples,
        population_size=1e4,
        recombination_rate=1e-8,
        sequence_length=2e5,
        random_seed=seed,
    )
    return msprime.sim_mutations(ts, rate=mutation_rate, random_seed=seed)


@pytest.fixture(scope="session")
def small_ts():
    """20 contemporary diploid samples, ~2e5 bp, with mutations."""
    return _small_ts(20, seed=11)


@pytest.fixture(scope="session")
def small_ts_unphased():
    """
    Like `small_ts` (15 diploids) with a high mutation rate so that there are many
    singletons; use with ``singletons_phased=False``. Samples belong to individuals.
    """
    return _small_ts(15, seed=12, mutation_rate=1e-7)


@pytest.fixture(scope="session")
def small_ts_historical():
    """15 diploid samples, 5 of them sampled at time 500 (non-contemporary)."""
    samples = [
        msprime.SampleSet(10, ploidy=2, time=0),
        msprime.SampleSet(5, ploidy=2, time=500),
    ]
    return _small_ts(samples, seed=13)
