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

from sigima.objects.signal import SignalObj
from sigima.proc.base import CombineToComplexParam
from sigima.proc.signal import combine_to_complex as combine_to_complex_signal
from sigima.tests.helpers import check_array_result
from sigima.tools.coordinates import polar_to_complex


@pytest.mark.validation
def test_signal_combine_to_complex() -> None:
    """Validation test for combine_to_complex (signal) function."""
    x = np.linspace(0, 1, 5)
    real = np.array([1, 2, 3, 4, 5])
    imag = np.array([10, 20, 30, 40, 50])
    mag = np.array([2, 3, 4, 5, 6])
    phase = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2, 0])

    # Test real_imag mode
    s_real = SignalObj("real")
    s_real.set_xydata(x, real)
    s_imag = SignalObj("imag")
    s_imag.set_xydata(x, imag)
    p = CombineToComplexParam()
    p.mode = "real_imag"
    result = combine_to_complex_signal(s_real, s_imag, p)
    check_array_result(
        "combine_to_complex_signal real_imag",
        result.y,
        real + 1j * imag,
    )

    # Test mag_phase mode (radians)
    s_mag = SignalObj("mag")
    s_mag.set_xydata(x, mag)
    s_phase = SignalObj("phase")
    s_phase.set_xydata(x, phase)
    p = CombineToComplexParam()
    p.mode = "mag_phase"
    p.unit = "rad"
    result = combine_to_complex_signal(s_mag, s_phase, p)
    check_array_result(
        "combine_to_complex_signal mag_phase",
        result.y,
        polar_to_complex(mag, phase, unit="rad"),
    )
