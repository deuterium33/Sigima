# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for signal analysis features
---------------------------------------

Features from the "Analysis" menu are covered by this test.
The "Analysis" menu contains functions to compute signal properties like
bandwidth, ENOB, etc.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

import numpy as np
import pytest

import sigima.objects
import sigima.params
import sigima.proc.signal
from sigima.tests import guiutils
from sigima.tests.data import get_test_signal
from sigima.tests.helpers import check_scalar_result


@pytest.mark.validation
def test_signal_bandwidth_3db() -> None:
    """Validation test for the bandwidth computation."""
    obj = get_test_signal("bandwidth.txt")
    geometry = sigima.proc.signal.bandwidth_3db(obj)
    assert geometry is not None, "Bandwidth computation failed."
    with guiutils.lazy_qt_app_context() as qt_app:
        if qt_app is not None:
            # pylint: disable=import-outside-toplevel
            from plotpy.builder import make

            from sigima.tests.vistools import view_curve_items

            x0, y0, x1, y1 = geometry.coords[0]
            x, y = obj.xydata
            view_curve_items(
                [
                    make.mcurve(x.real, y.real, label=obj.title),
                    make.annotated_segment(x0, y0, x1, y1, "Bandwidth@-3dB"),
                ],
                title="Bandwidth@-3dB",
            )
    length = geometry.segments_lengths()[0]
    check_scalar_result("Bandwidth@-3dB", length, 38.99301975103714)
    p1 = sigima.params.AbscissaParam.create(x=length)
    table = sigima.proc.signal.y_at_x(obj, p1)
    check_scalar_result("Value@cutoff", table["y@x"][0], np.max(obj.y) - 3.0)


@pytest.mark.validation
def test_dynamic_parameters() -> None:
    """Validation test for dynamic parameters computation."""
    obj = get_test_signal("dynamic_parameters.txt")
    param = sigima.params.DynamicParam.create(full_scale=1.0)
    table = sigima.proc.signal.dynamic_parameters(obj, param)
    assert table is not None, "Dynamic parameters computation failed"
    check_scalar_result("ENOB", table["enob"][0], 5.1, rtol=0.001)
    check_scalar_result("SINAD", table["sinad"][0], 32.49, rtol=0.001)
    check_scalar_result("THD", table["thd"][0], -30.18, rtol=0.001)
    check_scalar_result("SFDR", table["sfdr"][0], 34.03, rtol=0.001)
    check_scalar_result("Freq", table["freq"][0], 49998377.464, rtol=0.001)
    check_scalar_result("SNR", table["snr"][0], 101.52, rtol=0.001)


@pytest.mark.validation
def test_signal_sampling_rate_period() -> None:
    """Validation test for the sampling rate and period computation."""
    obj = get_test_signal("dynamic_parameters.txt")
    table = sigima.proc.signal.sampling_rate_period(obj)
    assert table is not None, "Sampling rate and period computation failed"
    check_scalar_result("Sampling rate", table["fs"][0], 1.0e10, rtol=0.001)
    check_scalar_result("Period", table["T"][0], 1.0e-10, rtol=0.001)


@pytest.mark.validation
def test_signal_contrast() -> None:
    """Validation test for the contrast computation."""
    obj = get_test_signal("fw1e2.txt")
    table = sigima.proc.signal.contrast(obj)
    assert table is not None, "Contrast computation failed"
    check_scalar_result("Contrast", table["contrast"][0], 0.825, rtol=0.001)


@pytest.mark.validation
def test_signal_x_at_minmax() -> None:
    """Validation test for the x value at min/max computation."""
    obj = get_test_signal("fw1e2.txt")
    table = sigima.proc.signal.x_at_minmax(obj)
    assert table is not None, "X at min/max computation failed"
    check_scalar_result("X@Ymin", table["X@Ymin"][0], 0.803, rtol=0.001)
    check_scalar_result("X@Ymax", table["X@Ymax"][0], 5.184, rtol=0.001)


@pytest.mark.validation
def test_signal_x_at_y() -> None:
    """Validation test for the abscissa finding computation."""
    obj = sigima.objects.create_signal_from_param(sigima.objects.StepParam.create())
    if obj is None:
        raise ValueError("Failed to create test signal")
    param = sigima.proc.signal.OrdinateParam.create(y=0.5)
    table = sigima.proc.signal.x_at_y(obj, param)
    assert table is not None, "X at Y computation failed"
    check_scalar_result("x|y=0.5", table["x@y"][0], 0.0)


@pytest.mark.validation
def test_signal_y_at_x() -> None:
    """Validation test for the ordinate finding computation."""
    param = sigima.objects.TriangleParam.create(xmin=0.0, xmax=10.0, size=101)
    obj = sigima.objects.create_signal_from_param(param)
    if obj is None:
        raise ValueError("Failed to create test signal")
    param = sigima.proc.signal.AbscissaParam.create(x=2.5)
    table = sigima.proc.signal.y_at_x(obj, param)
    assert table is not None, "Y at X computation failed"
    check_scalar_result("y|x=2.5", table["y@x"][0], 1.0)


if __name__ == "__main__":
    guiutils.enable_gui()
    test_signal_bandwidth_3db()
    test_dynamic_parameters()
    test_signal_sampling_rate_period()
    test_signal_contrast()
    test_signal_x_at_minmax()
    test_signal_x_at_y()
    test_signal_y_at_x()
