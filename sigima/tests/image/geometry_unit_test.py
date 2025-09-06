# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for geometry computation functions.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.ndimage as spi

import sigima.enums
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


@pytest.mark.validation
def test_image_resampling() -> None:
    """Image resampling test."""
    execenv.print("*** Testing image resampling")

    # Create a test image
    ima1 = get_test_image(
        "flower.npy"
    )  # Test 1: Identity resampling (same dimensions and coordinate range)
    p1 = sigima.params.Resampling2DParam.create(
        mode="shape",
        width=ima1.data.shape[1],
        height=ima1.data.shape[0],
        xmin=ima1.x0,
        xmax=ima1.x0 + ima1.width,
        ymin=ima1.y0,
        ymax=ima1.y0 + ima1.height,
        method=sigima.enums.Interpolation2DMethod.LINEAR,
    )
    dst1 = sigima.proc.image.resampling(ima1, p1)

    # Should be very close to original (allowing for small interpolation differences)
    check_scalar_result("Identity resampling X0", dst1.x0, ima1.x0)
    check_scalar_result("Identity resampling Y0", dst1.y0, ima1.y0)
    check_scalar_result(
        "Identity resampling shape[0]", dst1.data.shape[0], ima1.data.shape[0]
    )
    check_scalar_result(
        "Identity resampling shape[1]", dst1.data.shape[1], ima1.data.shape[1]
    )

    # Test 2: Downsample by factor of 2
    p2 = sigima.params.Resampling2DParam.create(
        mode="shape",
        width=ima1.data.shape[1] // 2,
        height=ima1.data.shape[0] // 2,
        xmin=ima1.x0,
        xmax=ima1.x0 + ima1.width,
        ymin=ima1.y0,
        ymax=ima1.y0 + ima1.height,
        method=sigima.enums.Interpolation2DMethod.LINEAR,
    )
    dst2 = sigima.proc.image.resampling(ima1, p2)

    check_scalar_result("Downsample X0", dst2.x0, ima1.x0)
    check_scalar_result("Downsample Y0", dst2.y0, ima1.y0)
    check_scalar_result(
        "Downsample shape[0]", dst2.data.shape[0], ima1.data.shape[0] // 2
    )
    check_scalar_result(
        "Downsample shape[1]", dst2.data.shape[1], ima1.data.shape[1] // 2
    )

    # Check that pixel sizes are adjusted correctly
    expected_dx = ima1.dx * 2 if ima1.dx is not None else 2.0
    expected_dy = ima1.dy * 2 if ima1.dy is not None else 2.0
    check_scalar_result("Downsample dx", dst2.dx, expected_dx, rtol=1e-10)
    check_scalar_result("Downsample dy", dst2.dy, expected_dy, rtol=1e-10)

    # Test 3: Use pixel size mode
    if ima1.dx is not None and ima1.dy is not None:
        p3 = sigima.params.Resampling2DParam.create(
            mode="dxy",
            dx=ima1.dx * 1.5,
            dy=ima1.dy * 1.5,
            xmin=ima1.x0,
            xmax=ima1.x0 + ima1.width,
            ymin=ima1.y0,
            ymax=ima1.y0 + ima1.height,
            method=sigima.enums.Interpolation2DMethod.LINEAR,
        )
        dst3 = sigima.proc.image.resampling(ima1, p3)

        check_scalar_result("Pixel size mode dx", dst3.dx, ima1.dx * 1.5, rtol=1e-10)
        check_scalar_result("Pixel size mode dy", dst3.dy, ima1.dy * 1.5, rtol=1e-10)

    # Test 4: Different interpolation methods
    for method in sigima.enums.Interpolation2DMethod:
        p4 = sigima.params.Resampling2DParam.create(
            mode="shape",
            width=ima1.data.shape[1] // 2,
            height=ima1.data.shape[0] // 2,
            xmin=ima1.x0,
            xmax=ima1.x0 + ima1.width,
            ymin=ima1.y0,
            ymax=ima1.y0 + ima1.height,
            method=method,
        )
        dst4 = sigima.proc.image.resampling(ima1, p4)

        # Basic shape checks
        check_scalar_result(
            f"Method {method} shape[0]", dst4.data.shape[0], ima1.data.shape[0] // 2
        )
        check_scalar_result(
            f"Method {method} shape[1]", dst4.data.shape[1], ima1.data.shape[1] // 2
        )

    # Test 5: fill_value parameter (out-of-bounds sampling)
    execenv.print("  Testing fill_value parameter")

    # Test 5a: Default behavior (fill_value=None should use NaN)
    p5a = sigima.params.Resampling2DParam.create(
        mode="shape",
        width=20,
        height=20,
        xmin=600.0,  # Outside image bounds
        xmax=620.0,
        ymin=600.0,
        ymax=620.0,
        method=sigima.enums.Interpolation2DMethod.LINEAR,
        fill_value=None,
    )
    dst5a = sigima.proc.image.resampling(ima1, p5a)

    # Should be all NaN since sampling outside image bounds
    assert np.all(np.isnan(dst5a.data)), (
        "Expected all NaN values for out-of-bounds sampling with fill_value=None"
    )
    assert dst5a.data.dtype == np.float64, "Expected float64 dtype for NaN result"

    # Test 5b: Custom fill value
    p5b = sigima.params.Resampling2DParam.create(
        mode="shape",
        width=20,
        height=20,
        xmin=600.0,  # Outside image bounds
        xmax=620.0,
        ymin=600.0,
        ymax=620.0,
        method=sigima.enums.Interpolation2DMethod.LINEAR,
        fill_value=123.0,
    )
    dst5b = sigima.proc.image.resampling(ima1, p5b)

    # Should be all 123.0 since sampling outside image bounds
    assert np.all(dst5b.data == 123.0), (
        "Expected all fill values for out-of-bounds sampling"
    )
    assert dst5b.data.dtype == ima1.data.dtype, (
        "Expected same dtype as input for numeric fill value"
    )

    # Test 5c: Partially outside (mix of real data and fill values)
    p5c = sigima.params.Resampling2DParam.create(
        mode="shape",
        width=30,
        height=30,
        xmin=ima1.x0 + ima1.width - 10,  # Partially outside
        xmax=ima1.x0 + ima1.width + 20,
        ymin=ima1.y0 + ima1.height - 10,
        ymax=ima1.y0 + ima1.height + 20,
        method=sigima.enums.Interpolation2DMethod.LINEAR,
        fill_value=99.0,
    )
    dst5c = sigima.proc.image.resampling(ima1, p5c)

    # Should have mix of values
    fill_count = np.sum(dst5c.data == 99.0)
    total_count = dst5c.data.size
    assert fill_count > 0, "Expected some fill values for partially out-of-bounds"
    assert fill_count < total_count, "Expected some real data values"

    # Test 5d: Within bounds should not use fill value
    p5d = sigima.params.Resampling2DParam.create(
        mode="shape",
        width=50,
        height=50,
        xmin=ima1.x0 + 50,  # Within bounds
        xmax=ima1.x0 + 100,
        ymin=ima1.y0 + 50,
        ymax=ima1.y0 + 100,
        method=sigima.enums.Interpolation2DMethod.LINEAR,
        fill_value=999.0,
    )
    dst5d = sigima.proc.image.resampling(ima1, p5d)

    # Should not contain any fill values since all within bounds
    assert not np.any(dst5d.data == 999.0), (
        "No fill values expected for within-bounds sampling"
    )


if __name__ == "__main__":
    test_image_fliph()
    test_image_flipv()
    test_image_rotate90()
    test_image_rotate270()
    test_image_rotate()
    test_image_transpose()
    test_image_resampling()
