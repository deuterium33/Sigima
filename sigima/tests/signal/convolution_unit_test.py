# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for signal convolution/deconvolution features."""

# pylint: disable=invalid-name

from __future__ import annotations

import numpy as np
import pytest

from sigima.objects import create_signal_from_param
from sigima.objects.signal import GaussParam, SignalObj, SinusParam, ZerosParam
from sigima.proc.signal import convolution, deconvolution
from sigima.tests import guiutils
from sigima.tests.data import get_test_signal
from sigima.tests.helpers import check_array_result
from sigima.tools.signal.fourier import deconvolve


def _generate_signal(size: int = 1024) -> SignalObj:
    """Generate a test signal.

    Args:
        size: The size of the signal to generate.

    Returns:
        A signal object.
    """
    # Sine wave signal.
    sine_wave = create_signal_from_param(
        SinusParam.create(
            size=size,
            xmin=-10.0,
            xmax=10.0,
            a=1.0,
            offset=0.0,
            freq=1.0,
        )
    )
    return sine_wave


@pytest.mark.validation
def test_signal_convolution() -> None:
    """Validation test for the signal convolution processing."""
    src1 = get_test_signal("paracetamol.txt")
    param = GaussParam.create(title="Gaussian", sigma=10.0)
    src2 = create_signal_from_param(param)
    dst = convolution(src1, src2)
    exp = np.convolve(src1.y, src2.y, mode="same")
    check_array_result("Convolution", dst.y, exp)


@pytest.mark.validation
def test_signal_deconvolution() -> None:
    """Validation test for signal deconvolution."""
    src = _generate_signal()
    # Identity kernel.
    kernel = create_signal_from_param(ZerosParam.create(xmin=0.0, xmax=1.0, size=256))
    kernel.y[0] = 1.0
    # Deconvolve the signal.
    deconvolved = deconvolution(src, kernel)
    # View the signals.
    guiutils.view_curves_if_gui([src, kernel, deconvolved], title="Deconvolution Test")
    # The deconvolution should be identical to the source.
    check_array_result("Deconvolution test", deconvolved.y, src.y)


def test_tools_signal_deconvolve_null_kernel() -> None:
    """Test deconvolution with a null kernel."""
    src = _generate_signal()
    xkernel = np.linspace(0, 1, 5)
    ykernel = np.zeros(5)  # All zeros.
    with pytest.raises(
        ValueError, match="Filter is all zeros, cannot be used to deconvolve."
    ):
        deconvolve(src.x, src.y, xkernel, ykernel)


def test_tools_signal_deconvolve_shape_error() -> None:
    """Test deconvolution with mismatched input shapes."""
    src = _generate_signal()
    xkernel = np.linspace(0, 1, 10)
    ykernel = np.ones(9)  # Mismatched kernel shape.
    with pytest.raises(
        ValueError, match="X data and Y data of the filter must have the same size."
    ):
        deconvolve(src.x, src.y, xkernel, ykernel)


if __name__ == "__main__":
    guiutils.enable_gui()
    test_signal_deconvolution()
    test_tools_signal_deconvolve_null_kernel()
    test_tools_signal_deconvolve_shape_error()
