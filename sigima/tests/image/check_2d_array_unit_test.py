# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for 2D-array function checks decorators."""

import numpy as np
import pytest

from sigima.tools.checks import check_2d_array


@check_2d_array(dtype=np.floating)
def identity(data: np.ndarray) -> np.ndarray:
    """Dummy image function returning input."""
    return data


def test_valid_2d_float_input() -> None:
    """Test with valid 2D float array."""
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    result = identity(data)
    np.testing.assert_array_equal(result, data)


def test_wrong_ndim() -> None:
    """Test with non-2D input."""
    data = np.array([1.0, 2.0, 3.0], dtype=float)  # 1D
    with pytest.raises(ValueError, match="Input array must be 2D"):
        identity(data)


def test_wrong_dtype() -> None:
    """Test with wrong dtype."""
    data = np.array([[1, 2], [3, 4]], dtype=int)
    with pytest.raises(
        TypeError, match="Input array must be of type <class 'numpy.floating'>"
    ):
        identity(data)


@check_2d_array(non_constant=True)
def normalize(data: np.ndarray) -> np.ndarray:
    """Normalize 2D array to range [0, 1]."""
    return data / np.nanmax(data)


def test_non_constant_error() -> None:
    """Test with non-constant input."""
    data = np.full((3, 3), 5.0)
    with pytest.raises(ValueError, match="Input array has no dynamic range."):
        normalize(data)


@check_2d_array(finite_only=True)
def sum_image(data: np.ndarray) -> float:
    """Sum all finite values in a 2D array."""
    return np.sum(data)


def test_non_finite_values() -> None:
    """Test with non-finite values."""
    data = np.array([[1.0, np.inf], [3.0, np.nan]])
    with pytest.raises(ValueError, match="Input array contains non-finite values."):
        sum_image(data)
