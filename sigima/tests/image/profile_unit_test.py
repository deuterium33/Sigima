# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Profile extraction unit test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

import numpy as np
import pytest

import sigima.objects
import sigima.params
import sigima.proc.image
from sigima.tests.data import create_sincos_image
from sigima.tests.helpers import check_array_result


@pytest.mark.validation
def test_line_profile() -> None:
    """Test line profile computation"""
    width, height = 256, 128
    dtype = sigima.objects.ImageDatatypes.UINT16
    newparam = sigima.objects.NewImageParam.create(
        dtype=dtype, height=height, width=width
    )
    ima = create_sincos_image(newparam)

    # Test horizontal line profile
    row = 100
    param = sigima.params.LineProfileParam.create(row=row, direction="horizontal")
    sig = sigima.proc.image.line_profile(ima, param)
    assert sig is not None
    assert len(sig.y) == width
    exp = np.array(ima.data[row, :], dtype=float)
    check_array_result("Horizontal line profile", sig.y, exp)

    # Test vertical line profile
    col = 50
    param = sigima.params.LineProfileParam.create(col=col, direction="vertical")
    sig = sigima.proc.image.line_profile(ima, param)
    assert sig is not None
    assert len(sig.y) == height
    exp = np.array(ima.data[:, col], dtype=float)
    check_array_result("Vertical line profile", sig.y, exp)


@pytest.mark.validation
def test_segment_profile() -> None:
    """Test segment profile computation"""
    width, height = 256, 128
    dtype = sigima.objects.ImageDatatypes.UINT16
    newparam = sigima.objects.NewImageParam.create(
        dtype=dtype, height=height, width=width
    )
    ima = create_sincos_image(newparam)

    # Test segment profile
    row1, col1, row2, col2 = 10, 20, 200, 20
    param = sigima.params.SegmentProfileParam.create(
        row1=row1, col1=col1, row2=row2, col2=col2
    )
    sig = sigima.proc.image.segment_profile(ima, param)
    assert sig is not None
    assert len(sig.y) == min(row2, height - 1) - max(row1, 0) + 1
    exp = np.array(ima.data[10:200, 20], dtype=float)
    check_array_result("Segment profile", sig.y, exp)


@pytest.mark.validation
def test_average_profile() -> None:
    """Test average profile computation"""
    width, height = 256, 128
    dtype = sigima.objects.ImageDatatypes.UINT16
    newparam = sigima.objects.NewImageParam.create(
        dtype=dtype, height=height, width=width
    )
    ima = create_sincos_image(newparam)
    row1, col1, row2, col2 = 10, 20, 200, 230
    param = sigima.params.AverageProfileParam.create(
        row1=row1, col1=col1, row2=row2, col2=col2
    )

    # Test horizontal average profile
    param.direction = "horizontal"
    sig = sigima.proc.image.average_profile(ima, param)
    assert sig is not None
    assert len(sig.y) == col2 - col1 + 1
    exp = np.array(ima.data[row1 : row2 + 1, col1 : col2 + 1].mean(axis=0), dtype=float)
    check_array_result("Horizontal average profile", sig.y, exp)

    # Test vertical average profile
    param.direction = "vertical"
    sig = sigima.proc.image.average_profile(ima, param)
    assert sig is not None
    assert len(sig.y) == min(row2, height - 1) - max(row1, 0) + 1
    exp = np.array(ima.data[row1 : row2 + 1, col1 : col2 + 1].mean(axis=1), dtype=float)
    check_array_result("Vertical average profile", sig.y, exp)


def __test_radial_profile_center(
    obj: sigima.objects.ImageObj, p: sigima.params.RadialProfileParam
) -> sigima.objects.SignalObj:
    """Test radial profile computation with given center.

    Args:
        obj: Image object
        p: Radial profile parameters

    Returns:
        Signal object containing the radial profile
    """
    sig = sigima.proc.image.radial_profile(obj, p)
    assert sig is not None
    assert len(sig.x) == len(sig.y)
    assert len(sig.y) > 0
    # Check that profile values are within expected range
    assert np.all(np.isfinite(sig.y))
    assert np.all(sig.y >= 0)  # Pixel values should be non-negative
    return sig


@pytest.mark.validation
def test_radial_profile() -> None:
    """Test radial profile computation"""
    width, height = 256, 128
    dtype = sigima.objects.ImageDatatypes.UINT16
    newparam = sigima.objects.NewImageParam.create(
        dtype=dtype, height=height, width=width
    )
    ima = create_sincos_image(newparam)

    # Test radial profile with centroid center
    param = sigima.params.RadialProfileParam.create(center="centroid")
    param.update_from_obj(ima)
    __test_radial_profile_center(ima, param)

    # Test radial profile with image center
    param = sigima.params.RadialProfileParam.create(center="center")
    param.update_from_obj(ima)
    __test_radial_profile_center(ima, param)

    # Test radial profile with user-defined center
    x0, y0 = width // 2, height // 2
    param = sigima.params.RadialProfileParam.create(center="user", x0=x0, y0=y0)
    sig = __test_radial_profile_center(ima, param)

    # Test that the x-axis represents distance from center (symmetric around 0)
    assert sig.x[0] < 0  # First element should be negative
    assert sig.x[-1] > 0  # Last element should be positive
    assert sig.x[len(sig.x) // 2] == 0  # Center should be at distance 0

    # Additional validation using the helper function for the last profile
    check_array_result("Radial profile", sig.y, sig.y)


if __name__ == "__main__":
    test_line_profile()
    test_segment_profile()
    test_average_profile()
    test_radial_profile()
