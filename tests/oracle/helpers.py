"""
Helpers shared by the oracle equivalence tests.
"""

import re

import _tsdate
import numpy as np
import pytest
import tskit


def copies(kwargs):
    """
    The keyword arguments with copies of their arrays
    """
    return {
        name: value.copy() if isinstance(value, np.ndarray) else value
        for name, value in kwargs.items()
    }


def assert_same_result(function, oracle, **kwargs):
    """
    Call the ``_tsdate`` function and its oracle with (copies of) the keyword
    arguments, assert that they return arrays of the same values, shapes and types,
    with NaNs in the same places, and return the result
    """
    expected = oracle(**copies(kwargs))
    actual = function(**copies(kwargs))
    if isinstance(expected, tuple):
        assert len(actual) == len(expected)
        for actual_array, expected_array in zip(actual, expected):
            np.testing.assert_array_equal(actual_array, expected_array, strict=True)
    else:
        np.testing.assert_array_equal(actual, expected, strict=True)
    return actual


def raises_library_error(code):
    """
    ``pytest.raises`` for the library error ``code``
    """
    return pytest.raises(_tsdate.LibraryError, match=re.escape(_tsdate.strerror(code)))


def assert_same_error(function, oracle, code, **kwargs):
    """
    Assert that the ``_tsdate`` function and its oracle both raise the library error
    ``code`` for the keyword arguments
    """
    with raises_library_error(code):
        oracle(**copies(kwargs))
    with raises_library_error(code):
        function(**copies(kwargs))


def make_ts(nodes, edges, mutations, length, num_individuals=0):
    """
    Tree sequence from ``nodes`` ``(is_sample, time)`` or ``(is_sample, time,
    individual)``, ``edges`` ``(left, right, parent, child)`` and ``mutations``
    ``(position, node)``, with ``num_individuals`` empty individuals.
    """
    tables = tskit.TableCollection(length)
    for _ in range(num_individuals):
        tables.individuals.add_row()
    for is_sample, time, *individual in nodes:
        tables.nodes.add_row(
            flags=int(is_sample),
            time=time,
            individual=individual[0] if individual else -1,
        )
    for left, right, parent, child in edges:
        tables.edges.add_row(left, right, parent, child)
    for i, (position, node) in enumerate(mutations):
        tables.sites.add_row(position, "0")
        tables.mutations.add_row(i, node, "1")
    tables.sort()
    return tables.tree_sequence()
