# tests/conftest.py
import pytest

def pytest_addoption(parser):
    """Registers the --run-gpu command-line option."""
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="Run tests that require GPU and are marked with skipif not config.getoption('--run-gpu')"
    ) 