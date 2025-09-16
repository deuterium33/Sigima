# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for I/O conversion functions
"""

from __future__ import annotations

import numpy as np
import pytest

from sigima.io.common import converters
from sigima.objects.image import ImageObj
from sigima.objects.signal import SignalObj


class TestConvertArrayToValidDType:
    """Test suite for convert_array_to_valid_dtype function."""

    def test_int_arrays(self) -> None:
        """Test conversion of integer numpy arrays."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.int32)
        result = converters.convert_array_to_valid_dtype(arr, SignalObj.VALID_DTYPES)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, arr, decimal=6)

        arr = np.array([[1, 2, 3], [1.1, 2, 3]], dtype=np.uint32)
        result = converters.convert_array_to_valid_dtype(arr, ImageObj.VALID_DTYPES)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        np.testing.assert_array_almost_equal(result, arr, decimal=6)

        arr = np.array([[1, 2, 3], [1.1, 2, 1e8]], dtype=np.uint32)
        result = converters.convert_array_to_valid_dtype(arr, ImageObj.VALID_DTYPES)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int32
        np.testing.assert_array_almost_equal(result, arr, decimal=6)

    def test_float_arrays(self) -> None:
        """Test conversion of float numpy arrays."""
        arr = np.array([1.1, 2.2, 3.3], dtype=np.float32)
        result = converters.convert_array_to_valid_dtype(arr, SignalObj.VALID_DTYPES)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, arr, decimal=6)

        arr = np.array([[1.1, 2.2, 3.3], [1.1, 2.2, 3.3]], dtype=np.float64)
        result = converters.convert_array_to_valid_dtype(arr, ImageObj.VALID_DTYPES)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        np.testing.assert_array_almost_equal(result, arr, decimal=6)

    def test_bool_arrays(self) -> None:
        """Test conversion of boolean numpy arrays."""
        arr = np.array([True, False, True], dtype=np.bool_)
        result = converters.convert_array_to_valid_dtype(arr, SignalObj.VALID_DTYPES)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8

    def test_empty_arrays(self) -> None:
        """Test conversion of empty numpy arrays."""
        arr = np.array([], dtype=np.float32)
        result = converters.convert_array_to_valid_dtype(arr, SignalObj.VALID_DTYPES)
        assert isinstance(result, np.ndarray)
        assert result.size == 0

    def test_invalid_input_type(self) -> None:
        """Test that invalid input types raise TypeError."""
        with pytest.raises(TypeError):
            converters.convert_array_to_valid_dtype(
                "not an array", SignalObj.VALID_DTYPES
            )


if __name__ == "__main__":
    pytest.main([__file__])
