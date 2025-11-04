# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for image projection functions."""

from __future__ import annotations

import numpy as np
import pytest

import sigima.objects
import sigima.proc.image
from sigima.tests import guiutils
from sigima.tests.data import create_sincos_image
from sigima.tests.helpers import check_array_result, check_scalar_result


@pytest.mark.validation
def test_image_horizontal_projection() -> None:
    """Test image horizontal projection."""
    width, height = 64, 48
    dtype = sigima.objects.ImageDatatypes.UINT16
    newparam = sigima.objects.NewImageParam.create(
        dtype=dtype, height=height, width=width
    )
    ima = create_sincos_image(newparam)

    # Add axis labels and units to the image.
    ima.xunit = "px"
    ima.yunit = "mm"
    ima.zunit = "a.u."
    ima.xlabel = "X position"
    ima.ylabel = "Y position"
    ima.zlabel = "Intensity"

    sig = sigima.proc.image.horizontal_projection(ima)
    assert sig is not None

    # Visualize image and result profile during interactive runs.
    guiutils.view_images_if_gui(ima, title="Horizontal projection test image")
    guiutils.view_curves_if_gui(sig, title="Horizontal projection profile")

    # Signal length should equal the number of columns.
    assert ima.data is not None
    assert len(sig.x) == ima.data.shape[1]
    # X-coordinates spacing should match the image's dx.
    dx = np.mean(np.diff(sig.x))
    assert ima.dx is not None
    check_scalar_result("X-axis spacing", dx, ima.dx)

    expected = np.sum(ima.data, axis=0, dtype=np.float64)
    check_array_result("Horizontal projection", sig.y, expected)

    # Check labels and units.
    assert sig.xlabel == ima.xlabel, (
        f"X-axis label mismatch: got {sig.xlabel}, expected {ima.xlabel}"
    )
    assert sig.xunit == ima.xunit, (
        f"X-axis unit mismatch: got {sig.xunit}, expected {ima.xunit}"
    )
    assert sig.ylabel == ima.zlabel, (
        f"Y-axis label mismatch: got {sig.ylabel}, expected {ima.zlabel}"
    )
    assert sig.yunit == ima.zunit, (
        f"Y-axis unit mismatch: got {sig.yunit}, expected {ima.zunit}"
    )


@pytest.mark.validation
def test_image_vertical_projection() -> None:
    """Test image vertical projection."""
    width, height = 128, 64
    dtype = sigima.objects.ImageDatatypes.UINT16
    newparam = sigima.objects.NewImageParam.create(
        dtype=dtype, height=height, width=width
    )
    ima = create_sincos_image(newparam)
    # Add axis labels and units to the image.
    ima.xunit = "px"
    ima.yunit = "mm"
    ima.zunit = "a.u."
    ima.xlabel = "X position"
    ima.ylabel = "Y position"
    ima.zlabel = "Intensity"

    sig = sigima.proc.image.vertical_projection(ima)
    assert sig is not None

    # Visualize image and result profile during interactive runs.
    guiutils.view_images_if_gui(ima, title="Vertical projection test image")
    guiutils.view_curves_if_gui(sig, title="Vertical projection profile")

    # Signal length should equal the number of rows.
    assert ima.data is not None
    assert len(sig.x) == ima.data.shape[0]
    # X-coordinates spacing should match the image's dy.
    dx = np.mean(np.diff(sig.x))
    assert ima.dy is not None
    check_scalar_result("X-axis spacing", dx, ima.dy)

    expected = np.sum(ima.data, axis=1, dtype=np.float64)
    check_array_result("Vertical projection", sig.y, expected)

    # Check labels and units.
    assert sig.xlabel == ima.ylabel, (
        f"X-axis label mismatch: got {sig.xlabel}, expected {ima.ylabel}"
    )
    assert sig.xunit == ima.yunit, (
        f"X-axis unit mismatch: got {sig.xunit}, expected {ima.yunit}"
    )
    assert sig.ylabel == ima.zlabel, (
        f"Y-axis label mismatch: got {sig.ylabel}, expected {ima.zlabel}"
    )
    assert sig.yunit == ima.zunit, (
        f"Y-axis unit mismatch: got {sig.yunit}, expected {ima.zunit}"
    )


if __name__ == "__main__":
    guiutils.enable_gui()
    test_image_horizontal_projection()
    test_image_vertical_projection()
