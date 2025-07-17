# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for coordinate conversion algorithms.

This module tests the conversion functions between polar, complex, and Cartesian
coordinates provided by `sigima_.algorithms.coordinates`.

Tested functions:
- polar_to_complex
- complex_to_polar
- complex_to_cartesian
"""

import numpy as np
import pytest

from sigima.tests.helpers import check_array_result
from sigima.tools.coordinates import polar_to_complex


def test_polar_to_complex_rad():
    """Test polar_to_complex with radians."""
    r = np.array([1, 2])
    theta = np.array([0, np.pi / 2])
    z = polar_to_complex(r, theta, unit="rad")
    check_array_result("polar_to_complex_rad", z, np.array([1 + 0j, 0 + 2j]))


def test_polar_to_complex_deg():
    """Test polar_to_complex with degrees."""
    r = np.array([1, 2])
    theta = np.array([0, 90])
    z = polar_to_complex(r, theta, unit="deg")
    check_array_result("polar_to_complex_deg", z, np.array([1 + 0j, 0 + 2j]))


def test_invalid_unit_polar_to_complex():
    """Test polar_to_complex with invalid unit raises ValueError."""
    r = np.array([1])
    theta = np.array([0])
    with pytest.raises(ValueError):
        polar_to_complex(r, theta, unit="foo")
