# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for geometry computation functions.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.ndimage as spi

import sigima.objects
import sigima.params
import sigima.proc.image
from sigima.tests.data import get_test_image, iterate_noisy_images
from sigima.tests.env import execenv
from sigima.tests.helpers import check_array_result, check_scalar_result


@pytest.mark.validation
def test_image_translate() -> None:
    """Image translation test."""
    for dx, dy in [(10, 0), (0, 10), (-10, -10)]:
        compfunc = sigima.proc.image.translate
        execenv.print(f"*** Testing image translate: {compfunc.__name__}")
        ima1 = list(iterate_noisy_images(size=128))[0]
        ima2: sigima.objects.ImageObj = compfunc(
            ima1, sigima.params.TranslateParam.create(dx=dx, dy=dy)
        )
        check_scalar_result("Image X translation", ima2.x0, ima1.x0 + dx)
        check_scalar_result("Image Y translation", ima2.y0, ima1.y0 + dy)


def __generic_flip_check(compfunc: callable, expfunc: callable) -> None:
    """Generic flip check function."""
    execenv.print(f"*** Testing image flip: {compfunc.__name__}")
    for ima1 in iterate_noisy_images(size=128):
        execenv.print(f"  {compfunc.__name__}({ima1.data.dtype}): ", end="")
        ima2: sigima.objects.ImageObj = compfunc(ima1)
        check_array_result("Image flip", ima2.data, expfunc(ima1.data))


@pytest.mark.validation
def test_image_fliph() -> None:
    """Image horizontal flip test."""
    __generic_flip_check(sigima.proc.image.fliph, np.fliplr)


@pytest.mark.validation
def test_image_flipv() -> None:
    """Image vertical flip test."""
    __generic_flip_check(sigima.proc.image.flipv, np.flipud)


def __generic_rotate_check(angle: int) -> None:
    """Generic rotate check function."""
    execenv.print(f"*** Testing image {angle}° rotation:")
    for ima1 in iterate_noisy_images(size=128):
        execenv.print(f"  rotate{angle}({ima1.data.dtype}): ", end="")
        ima2 = getattr(sigima.proc.image, f"rotate{angle}")(ima1)
        check_array_result(
            f"Image rotate{angle}", ima2.data, np.rot90(ima1.data, k=angle // 90)
        )


@pytest.mark.validation
def test_image_rotate90() -> None:
    """Image 90° rotation test."""
    __generic_rotate_check(90)


@pytest.mark.validation
def test_image_rotate270() -> None:
    """Image 270° rotation test."""
    __generic_rotate_check(270)


def __get_test_image_with_roi() -> sigima.objects.ImageObj:
    """Get a test image with a predefined ROI."""
    ima = get_test_image("flower.npy")
    ima.roi = sigima.objects.create_image_roi(
        "rectangle", [10.0, 10.0, 50.0, 400.0], indices=False
    )
    return ima


def __check_roi_properties(
    ima1: sigima.objects.ImageObj, ima2: sigima.objects.ImageObj
) -> None:
    """Check that the ROI properties are preserved after transformation."""
    assert ima2.roi.single_rois[0].title == ima1.roi.single_rois[0].title
    assert ima2.roi.single_rois[0].indices == ima1.roi.single_rois[0].indices


def test_roi_rotate90() -> None:
    """Test 90° rotation with ROI transformation."""
    ima = __get_test_image_with_roi()

    # Apply 90° rotation
    rotated = sigima.proc.image.rotate90(ima)

    # Check that ROI coordinates were transformed correctly
    # Original: [10, 10, 50, 400] -> Expected: [10, ima.height - 10 - 50, 400, 50]
    expected_coords = np.array([10.0, ima.height - 60.0, 400.0, 50.0])
    actual_coords = rotated.roi.single_rois[0].coords

    assert np.allclose(actual_coords, expected_coords), (
        f"ROI coordinates not transformed correctly. "
        f"Expected {expected_coords}, got {actual_coords}"
    )
    __check_roi_properties(ima, rotated)


def test_roi_rotate270() -> None:
    """Test 270° rotation with ROI transformation."""
    ima = __get_test_image_with_roi()

    # Apply 270° rotation
    rotated = sigima.proc.image.rotate270(ima)

    # Check that ROI coordinates were transformed correctly
    # Original: [10, 10, 50, 400] -> Expected: [ima.width - 10 - 400, 10, 400, 50]
    expected_coords = np.array([ima.width - 410.0, 10.0, 400.0, 50.0])
    actual_coords = rotated.roi.single_rois[0].coords

    assert np.allclose(actual_coords, expected_coords), (
        f"ROI coordinates not transformed correctly. "
        f"Expected {expected_coords}, got {actual_coords}"
    )
    __check_roi_properties(ima, rotated)


def test_roi_translation() -> None:
    """Test translation with ROI transformation."""
    ima = __get_test_image_with_roi()

    # Apply translation
    translated = sigima.proc.image.translate(
        ima, sigima.params.TranslateParam.create(dx=10, dy=10)
    )

    # Check that ROI coordinates were transformed correctly
    # Original: [10, 10, 50, 400] -> Expected: [20, 20, 50, 400]
    expected_coords = np.array([20.0, 20.0, 50.0, 400.0])
    actual_coords = translated.roi.single_rois[0].coords

    assert np.allclose(actual_coords, expected_coords), (
        f"ROI coordinates not transformed correctly. "
        f"Expected {expected_coords}, got {actual_coords}"
    )
    __check_roi_properties(ima, translated)


@pytest.mark.validation
def test_image_rotate() -> None:
    """Image rotation test."""
    execenv.print("*** Testing image rotation:")
    for ima1 in iterate_noisy_images(size=128):
        for angle in (30.0, 45.0, 60.0, 120.0):
            execenv.print(f"  rotate{angle}({ima1.data.dtype}): ", end="")
            ima2 = sigima.proc.image.rotate(
                ima1, sigima.params.RotateParam.create(angle=angle)
            )
            exp = spi.rotate(ima1.data, angle, reshape=False)
            check_array_result(f"Image rotate{angle}", ima2.data, exp)


@pytest.mark.validation
def test_image_transpose() -> None:
    """Validation test for the image transpose processing."""
    src = get_test_image("flower.npy")
    dst = sigima.proc.image.transpose(src)
    exp = np.swapaxes(src.data, 0, 1)
    check_array_result("Transpose", dst.data, exp)


if __name__ == "__main__":
    test_image_fliph()
    test_image_flipv()
    test_image_rotate90()
    test_image_rotate270()
    test_image_rotate()
    test_image_transpose()
