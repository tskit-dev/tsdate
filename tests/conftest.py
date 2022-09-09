import pytest


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
