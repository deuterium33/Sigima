# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.
"""
Unit tests for image measurement computations.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy import ma

import sigima.objects
import sigima.proc.image as sp_image
from sigima.proc.image.measurement import SumPixelsAlongAxisParam
from sigima.tests import guiutils
from sigima.tests.data import create_sincos_image
from sigima.tests.helpers import check_array_result


@pytest.mark.validation
def test_image_sum_pixels_along_x_axis() -> None:
    """Test summing pixels along X (sum across columns -> result indexed by rows)."""
    width, height = 128, 64
    dtype = sigima.objects.ImageDatatypes.UINT16
    newparam = sigima.objects.NewImageParam.create(
        dtype=dtype, height=height, width=width
    )
    ima = create_sincos_image(newparam)

    # Add explicit spatial units and axis labels to the image
    ima.xunit = "px"
    ima.yunit = "mm"
    ima.zunit = "a.u."
    ima.xlabel = "X position"
    ima.ylabel = "Y position"
    ima.zlabel = "Intensity (a.u.)"

    param = SumPixelsAlongAxisParam.create(axis="X")
    sig = sp_image.sum_pixels_along_axis(ima, param)
    assert sig is not None

    # Y-array length should match number of rows
    assert len(sig.y) == ima.data.shape[0]

    # Expected = sum over axis=1 (columns)
    expected = np.asarray(np.sum(ima.data, axis=1), dtype=float)
    check_array_result("Sum along X (rows)", sig.y, expected)

    # Check units/labels: summing along X returns a profile indexed by Y
    assert sig.xunit == ima.yunit, (
        f"Y-axis unit mismatch: got {sig.xunit}, expected {ima.yunit}"
    )
    assert sig.yunit == ima.zunit, (
        f"Z-axis unit mismatch: got {sig.yunit}, expected {ima.zunit}"
    )
    assert sig.xlabel == ima.ylabel, (
        f"Y-axis label mismatch: got {sig.xlabel}, expected {ima.ylabel}"
    )
    assert sig.ylabel == ima.zlabel, (
        f"Z-axis label mismatch: got {sig.ylabel}, expected {ima.zlabel}"
    )

    # X coordinates spacing should match image.dy and be centered (mean ~ 0)
    dxs = np.diff(sig.x)
    assert np.allclose(dxs, ima.dy, rtol=1e-6, atol=1e-9), (
        f"Y-axis spacing mismatch: got {dxs}, expected {ima.dy}"
    )
    assert np.allclose(np.mean(sig.x), 0.0, atol=1e-9), (
        f"Y-axis coordinates not centered: mean={np.mean(sig.x)}"
    )

    # Visualize image and result curve for manual inspection during validation runs
    guiutils.view_images_if_gui(
        ima, title=f"Test image for {test_image_sum_pixels_along_x_axis.__name__}"
    )
    guiutils.view_curves_if_gui(
        sig,
        title=f"Summed profile (axis=X) for"
        f" {test_image_sum_pixels_along_x_axis.__name__}",
    )


@pytest.mark.validation
def test_image_sum_pixels_along_axis() -> None:
    """Test summing pixels along Y (sum across rows -> result indexed by columns),
    and ensure masked columns produce NaN in the output signal.
    """
    width, height = 64, 48
    dtype = sigima.objects.ImageDatatypes.UINT16
    newparam = sigima.objects.NewImageParam.create(
        dtype=dtype, height=height, width=width
    )
    ima = create_sincos_image(newparam)

    # Add explicit spatial units and axis labels to the image
    ima.xunit = "um"
    ima.yunit = "px"
    ima.zunit = "counts"
    ima.xlabel = "X [um]"
    ima.ylabel = "Y [px]"
    ima.zlabel = "Counts"

    # Convert to masked array and mask an entire column
    ima.data = ma.array(ima.data)
    col_to_mask = 5
    ima.data.mask = np.zeros_like(ima.data, dtype=bool)
    ima.data.mask[:, col_to_mask] = True

    param = SumPixelsAlongAxisParam.create(axis="Y")
    sig = sp_image.sum_pixels_along_axis(ima, param)
    assert sig is not None

    # Y-array length should match number of columns
    assert len(sig.y) == ima.data.shape[1]

    # Expected using masked-aware sum -> masked entries become NaN after filling
    expected = np.asarray(ma.filled(ma.sum(ima.data, axis=0).astype(float), np.nan))

    # Masked column should produce NaN in both expected and computed signal
    assert np.isnan(expected[col_to_mask])
    assert np.isnan(sig.y[col_to_mask])

    # Compare non-NaN entries
    valid = ~np.isnan(expected)
    np.testing.assert_allclose(sig.y[valid], expected[valid], rtol=1e-6, atol=1e-9)

    # Check units/labels: summing along Y returns a profile indexed by X
    assert sig.xunit == ima.xunit, (
        f"X-axis unit mismatch: got {sig.xunit}, expected {ima.xunit}"
    )
    assert sig.yunit == ima.zunit, (
        f"Z-axis unit mismatch: got {sig.yunit}, expected {ima.zunit}"
    )
    assert sig.xlabel == ima.xlabel, (
        f"X-axis label mismatch: got {sig.xlabel}, expected {ima.xlabel}"
    )
    assert sig.ylabel == ima.zlabel, (
        f"Z-axis label mismatch: got {sig.ylabel}, expected {ima.zlabel}"
    )

    # X coordinates spacing should match image.dx and be centered
    dxs = np.diff(sig.x)
    assert np.allclose(dxs, ima.dx, rtol=1e-6, atol=1e-9), (
        f"X-axis spacing mismatch: got {dxs}, expected {ima.dx}"
    )
    assert np.allclose(np.mean(sig.x), 0.0, atol=1e-9), (
        f"X-axis coordinates not centered: mean={np.mean(sig.x)}"
    )

    # Visualize masked image and summed profile for manual inspection
    # during validation runs
    guiutils.view_images_if_gui(
        ima,
        title=f"Masked test image for {test_image_sum_pixels_along_axis.__name__}",
    )
    guiutils.view_curves_if_gui(
        sig,
        title=f"Summed profile (axis=Y) for "
        f"{test_image_sum_pixels_along_axis.__name__}",
    )


if __name__ == "__main__":
    guiutils.enable_gui()
    test_image_sum_pixels_along_x_axis()
    test_image_sum_pixels_along_axis()
