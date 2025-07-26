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
from sigima.tests.helpers import check_array_result


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
def test_image_flipd() -> None:
    """Image diagonal flip test."""
    __generic_flip_check(sigima.proc.image.swap_axes, np.transpose)


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


@pytest.mark.validation
def test_image_rotate() -> None:
    """Image rotation test."""
    execenv.print("*** Testing image rotation:")
    for ima1 in iterate_noisy_images(size=128):
        for angle in (30, 45, 60, 120):
            execenv.print(f"  rotate{angle}({ima1.data.dtype}): ", end="")
            ima2 = sigima.proc.image.rotate(
                ima1, sigima.params.RotateParam.create(angle=angle)
            )
            exp = spi.rotate(ima1.data, angle, reshape=False)
            check_array_result(f"Image rotate{angle}", ima2.data, exp)


@pytest.mark.validation
def test_image_swap_axes() -> None:
    """Validation test for the image axes swapping processing."""
    src = get_test_image("flower.npy")
    dst = sigima.proc.image.swap_axes(src)
    exp = np.swapaxes(src.data, 0, 1)
    check_array_result("SwapAxes", dst.data, exp)


if __name__ == "__main__":
    test_image_fliph()
    test_image_flipv()
    test_image_rotate90()
    test_image_rotate270()
    test_image_rotate()
    test_image_swap_axes()
