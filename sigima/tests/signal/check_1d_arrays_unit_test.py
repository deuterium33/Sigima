# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for 1D-array function checks decorators."""

import numpy as np
import pytest

from sigima.tools.checks import check_1d_arrays


@check_1d_arrays(y_dtype=np.floating)
def add_arrays(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Dummy function to demonstrate basic signal requirements.

    This function takes two 1-D arrays of floats and returns their sum.

    Args:
        x: 1-D array of floats.
        y: 1-D array of floats.

    Returns:
        Result of the operation.
    """
    return x + y


def test_valid_input():
    """Test with valid 1-D float arrays of the same size."""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    result = add_arrays(x, y)
    np.testing.assert_array_equal(result, x + y)


def test_invalid_x():
    """Test invalid x: not 1-D, not float, or not sorted in ascending order."""
    y = np.array([4.0, 5.0, 6.0], dtype=float)

    x1 = np.array([[1.0, 2.0], [3.0, 4.0]])  # not 1-D
    with pytest.raises(ValueError, match="x must be 1-D."):
        add_arrays(x1, y)

    x2 = np.array([1, 2, 3], dtype=int)  # not float
    with pytest.raises(TypeError, match="x must be of type <class 'numpy.floating'>."):
        add_arrays(x2, y)


def test_invalid_y():
    """Test invalid y: not 1-D or not float."""
    x = np.array([1.0, 2.0, 3.0], dtype=float)

    y1 = np.array([[4.0, 5.0], [6.0, 7.0]])  # not 1-D
    with pytest.raises(ValueError, match="y must be 1-D."):
        add_arrays(x, y1)

    y2 = np.array([4, 5, 6], dtype=int)  # not float
    with pytest.raises(TypeError, match="y must be of type <class 'numpy.floating'>."):
        add_arrays(x, y2)


def test_size_mismatch():
    """Test x and y with different sizes."""
    x = np.array([1.0, 2.0, 3.0], dtype=float)
    y = np.array([4.0, 5.0], dtype=float)
    with pytest.raises(ValueError, match="x and y must have the same size."):
        add_arrays(x, y)


@check_1d_arrays(x_sorted=True, x_evenly_spaced=True)
def multiply_arrays(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Dummy function to demonstrate evenly spaced check.

    This function takes a 1-D array of floats and returns it multiplied by 2.

    Args:
        x: 1-D array of floats.
        y: 1-D array of floats.

    Returns:
        Result of the operation.
    """
    return x * y


def test_evenly_spaced_and_sorted():
    """Test with evenly spaced x."""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    result = multiply_arrays(x, y)
    np.testing.assert_array_equal(result, x * y)


def test_single_element_evenly_spaced():
    """Test with a single-element x."""
    x = np.array([42.0], dtype=np.float64)
    y = np.array([4.0], dtype=np.float64)
    result = multiply_arrays(x, y)
    np.testing.assert_array_equal(result, x * y)


def test_not_evenly_spaced():
    """Test with x that is not evenly spaced."""
    x = np.array([0.0, 1.0, 3.0], dtype=float)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    with pytest.raises(ValueError, match="x must be evenly spaced."):
        multiply_arrays(x, y)


def test_not_sorted():
    """Test with x that is not sorted in ascending order."""
    x = np.array([3.0, 1.0, 2.0], dtype=float)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    with pytest.raises(ValueError, match="x must be sorted in ascending order."):
        multiply_arrays(x, y)


@check_1d_arrays(x_evenly_spaced=True, rtol=1e-2)
def add_arrays_with_tolerant_spacing(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Dummy function accepting nearly-evenly spaced x.

    Args:
        x: 1-D array of floats.
        y: 1-D array of floats.

    Returns:
        Result of the operation.
    """
    return x + y


@check_1d_arrays(x_evenly_spaced=True)
def add_arrays_without_tolerance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Dummy function for evenly spaced x.

    Args:
        x: 1-D array of floats.
        y: 1-D array of floats.

    Returns:
        Result of the operation.
    """
    return x + y


def test_tolerant_spacing():
    """Test x array that is almost evenly spaced with relaxed tolerance."""
    # The spacing varies slightly (1 +/- 0.01)
    x = np.array([0.0, 1.0, 2.01], dtype=float)
    y = np.array([1.0, 1.0, 1.0], dtype=float)

    # Should pass with rtol=1e-2
    result = add_arrays_with_tolerant_spacing(x, y)
    np.testing.assert_array_equal(result, x + y)

    # Should fail with decorator's default value (rtol=1e-5)
    with pytest.raises(ValueError, match="x must be evenly spaced."):
        add_arrays_without_tolerance(x, y)


if __name__ == "__main__":
    test_valid_input()
    test_invalid_x()
    test_invalid_y()
    test_size_mismatch()
    test_evenly_spaced_and_sorted()
    test_single_element_evenly_spaced()
    test_not_evenly_spaced()
    test_not_sorted()
    test_tolerant_spacing()
