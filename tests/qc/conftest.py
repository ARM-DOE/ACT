import pytest


def pytest_addoption(parser):
    parser.addoption("--runbig", action="store_true", default=False, help="Run big tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "big: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runbig"):
        # --runbig given in cli: do not skip big tests
        return
    skip_big = pytest.mark.skip(reason="need --runbig option to run")
    for item in items:
        if "big" in item.keywords:
            item.add_marker(skip_big)
