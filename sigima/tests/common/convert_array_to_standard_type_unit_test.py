from __future__ import annotations

import numpy as np
import pytest

from sigima.io.common.converters import convert_array_to_valid_dtype
from sigima.objects.image import ImageObj
from sigima.objects.signal import SignalObj

# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for convert_array_to_standard_type function.
"""


def test_convert_array_to_standard_type_int() -> None:
    """Test conversion of float numpy array to standard type."""
    arr = np.array([1.0, 2.0, 3.0], dtype=np.int32)
    result = convert_array_to_valid_dtype(arr, SignalObj)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    np.testing.assert_array_almost_equal(result, arr, decimal=6)

    arr = np.array([[1, 2, 3], [1.1, 2, 3]], dtype=np.uint32)
    result = convert_array_to_valid_dtype(arr, ImageObj)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    np.testing.assert_array_almost_equal(result, arr, decimal=6)

    arr = np.array([[1, 2, 3], [1.1, 2, 1e8]], dtype=np.uint32)
    result = convert_array_to_valid_dtype(arr, ImageObj)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int32
    np.testing.assert_array_almost_equal(result, arr, decimal=6)


def test_convert_array_to_standard_type_float() -> None:
    """Test conversion of float numpy array to standard type."""
    arr = np.array([1.1, 2.2, 3.3], dtype=np.float32)
    result = convert_array_to_valid_dtype(arr, SignalObj)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    np.testing.assert_array_almost_equal(result, arr, decimal=6)

    arr = np.array([[1.1, 2.2, 3.3], [1.1, 2.2, 3.3]], dtype=np.float64)
    result = convert_array_to_valid_dtype(arr, ImageObj)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    np.testing.assert_array_almost_equal(result, arr, decimal=6)


def test_convert_array_to_standard_type_bool() -> None:
    """Test conversion of boolean numpy array to standard type."""
    arr = np.array([True, False, True], dtype=np.bool_)
    result = convert_array_to_valid_dtype(arr, SignalObj)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8


def test_convert_array_to_standard_type_empty() -> None:
    """Test conversion of empty numpy array."""
    arr = np.array([], dtype=np.float32)
    result = convert_array_to_valid_dtype(arr, SignalObj)
    assert isinstance(result, np.ndarray)
    assert result.size == 0


def test_convert_array_to_standard_type_invalid_type() -> None:
    """Test conversion raises TypeError for invalid input."""
    with pytest.raises(TypeError):
        convert_array_to_valid_dtype("not an array", SignalObj)  # type: ignore


if __name__ == "__main__":
    test_convert_array_to_standard_type_int()
    test_convert_array_to_standard_type_float()
    test_convert_array_to_standard_type_bool()
    test_convert_array_to_standard_type_empty()
    test_convert_array_to_standard_type_invalid_type()
