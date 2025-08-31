# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for :py:mod:`sigima.tools.coordinates`.

This module verifies the correctness of coordinate conversion functions.
"""

import numpy as np
import pytest

from sigima.tests.helpers import check_array_result
from sigima.tools.coordinates import polar_to_complex

polar_to_complex_parameters = [
    (
        np.array([1.0, 2.0, 3.0]),
        np.array([0.0, 90.0, 180.0]),
        "°",
        np.array([1 + 0j, 0 + 2j, -3 + 0j]),
    ),
    (
        np.array([1.0, 2.0, 3.0]),
        np.array([0.0, np.pi / 2, np.pi]),
        "rad",
        np.array([1 + 0j, 0 + 2j, -3 + 0j]),
    ),
]


@pytest.mark.parametrize("r, theta, unit, expected", polar_to_complex_parameters)
def test_polar_to_complex(r, theta, unit, expected):
    """Test :py:func:`sigima.tools.polar_to_complex` with valid input.

    Args:
        r: The radial coordinates.
        theta: The angular coordinates.
        unit: The unit of the angular coordinates ("°" or "rad").
        expected: The expected complex coordinates.
    """
    z = polar_to_complex(r, theta, unit=unit)
    check_array_result(f"polar_to_complex_{unit}", z, expected)


def test_polar_to_complex_invalid_unit():
    """Test :py:func:`sigima.tools.polar_to_complex` with invalid unit.

    Ensure :py:func:`sigima.tools.polar_to_complex` raises ValueError for unsupported
    unit.
    """
    r = np.array([1.0])
    theta = np.array([0.0])
    with pytest.raises(ValueError):
        polar_to_complex(r, theta, unit="foo")


if __name__ == "__main__":
    for parameters in polar_to_complex_parameters:
        test_polar_to_complex(*parameters)
    test_polar_to_complex_invalid_unit()
