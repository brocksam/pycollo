"""Fixtures and utilities for integration tests."""


import pytest


@pytest.fixture(scope="module")
def state():
    """Fixture instance to hold the (incremental) test state."""
    class State:
        pass
    return State()
