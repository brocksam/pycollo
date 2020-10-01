"""Generic pytest-related classes/functions used throughout test suite."""


import pytest


# Store history of failures per test class name and per index in parametrize
# (if parametrize used)
_test_failed_incremental = {}


def pytest_runtest_makereport(item, call):
    """Allow pytest to run tests as incremental.

    Sucessive, related tests are marked as xfailed if a previous test fails.
    Taken from "https://docs.pytest.org/en/latest/example/simple.html"

    """
    if "incremental" in item.keywords:
        # incremental marker is used
        if call.excinfo is not None:
            # the test has failed
            # retrieve the class name of the test
            cls_name = str(item.cls)
            # retrieve the index of the test (if parametrize is used in
            # combination with incremental)
            parametrize_index = (
                tuple(item.callspec.indices.values())
                if hasattr(item, "callspec")
                else ()
            )
            # retrieve the name of the test function
            test_name = item.originalname or item.name
            # store in _test_failed_incremental the original name of the failed
            # test
            _test_failed_incremental.setdefault(cls_name, {}).setdefault(
                parametrize_index, test_name
            )


def pytest_runtest_setup(item):
    """Allow pytest to run tests as incremental.

    Sucessive, related tests are marked as xfailed if a previous test fails.
    Taken from "https://docs.pytest.org/en/latest/example/simple.html"

    """
    if "incremental" in item.keywords:
        # retrieve the class name of the test
        cls_name = str(item.cls)
        # check if a previous test has failed for this class
        if cls_name in _test_failed_incremental:
            # retrieve the index of the test (if parametrize is used in
            # combination with incremental)
            parametrize_index = (
                tuple(item.callspec.indices.values())
                if hasattr(item, "callspec")
                else ()
            )
            # retrieve the name of the first test function to fail for this
            # class name and index
            test_class = _test_failed_incremental[cls_name]
            test_name = test_class.get(parametrize_index, None)
            # if name found, test has failed for the combination of class name
            # & test name
            if test_name is not None:
                pytest.xfail("previous test failed ({})".format(test_name))
