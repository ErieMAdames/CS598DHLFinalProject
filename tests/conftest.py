"""Pytest configuration for shared fixtures and CLI options."""

import os

import pytest

DATA_ROOT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "dreamt",
    "data_64Hz",
)


def pytest_addoption(parser):
    parser.addoption(
        "--real-data",
        action="store_true",
        default=False,
        help="Run tests against the real DREAMT dataset",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_data: mark test as requiring real DREAMT data",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--real-data"):
        return
    skip = pytest.mark.skip(reason="needs --real-data flag")
    for item in items:
        if "real_data" in item.keywords:
            item.add_marker(skip)
