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

from sigima.objects.image import ImageObj
from sigima.proc.base import CombineToComplexParam
from sigima.proc.image.mathops import combine_to_complex as combine_to_complex_image
from sigima.tests.helpers import check_array_result
from sigima.tools.coordinates import polar_to_complex


@pytest.mark.validation
def test_image_combine_to_complex() -> None:
    """Validation test for combine_to_complex (image) function."""
    x = np.linspace(0, 1, 4)
    y = np.linspace(0, 1, 4)
    real = np.ones((4, 4))
    imag = np.arange(16).reshape(4, 4)
    mag = np.full((4, 4), 2.0)
    phase = np.linspace(0, np.pi, 16).reshape(4, 4)

    # Test real_imag mode
    img_real = ImageObj("real")
    img_real.x, img_real.y, img_real.z = x, y, real
    img_imag = ImageObj("imag")
    img_imag.x, img_imag.y, img_imag.z = x, y, imag
    p = CombineToComplexParam()
    p.mode = "real_imag"
    result = combine_to_complex_image(img_real, img_imag, p)
    check_array_result(
        "combine_to_complex_image real_imag",
        result.z,
        real + 1j * imag,
    )

    # Test mag_phase mode (radians)
    img_mag = ImageObj("mag")
    img_mag.x, img_mag.y, img_mag.z = x, y, mag
    img_phase = ImageObj("phase")
    img_phase.x, img_phase.y, img_phase.z = x, y, phase
    p = CombineToComplexParam()
    p.mode = "mag_phase"
    p.unit = "rad"
    result = combine_to_complex_image(img_mag, img_phase, p)
    check_array_result(
        "combine_to_complex_image mag_phase",
        result.z,
        polar_to_complex(mag, phase, unit="rad"),
    )
