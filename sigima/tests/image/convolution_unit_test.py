# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for image convolution/deconvolution features."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.signal as sps

from sigima.config import options as sigima_options
from sigima.objects import create_image, create_image_from_param
from sigima.objects.image import Gauss2DParam, ImageObj, Zero2DParam
from sigima.proc.image.mathops import convolution, deconvolution
from sigima.tests import guiutils
from sigima.tests.helpers import check_array_result
from sigima.tools.image import deconvolve


@pytest.fixture
def disable_kernel_normalization():
    """Temporarily disable kernel normalization for tests."""
    # Save current values
    original_auto_normalize = sigima_options.auto_normalize_kernel.get()
    original_warn = sigima_options.warn_unnormalized_kernel.get()

    # Disable for test
    sigima_options.auto_normalize_kernel.set(False)
    sigima_options.warn_unnormalized_kernel.set(False)

    yield

    # Restore original values
    sigima_options.auto_normalize_kernel.set(original_auto_normalize)
    sigima_options.warn_unnormalized_kernel.set(original_warn)


def _generate_rectangle_image(title: str = "Rectangle", size: int = 32) -> ImageObj:
    """Generate a test square image with a rectangle in the center."""
    data = np.zeros((size, size), dtype=np.float64)
    data[size // 5 : 2 * size // 5, size // 7 : 5 * size // 7] = 1.0
    img = create_image(title, data)
    return img


def _generate_image(size: int = 32) -> ImageObj:
    """Generate a test square image.

    Args:
        size: The dimension of the square image to generate.

    Returns:
        An image object.
    """
    # Gaussian image.
    gauss_img = create_image_from_param(Gauss2DParam.create(height=size, width=size))
    return gauss_img


def _generate_gaussian_kernel(size: int = 32, sigma: float = 1.0) -> ImageObj:
    """Generate a Gaussian kernel image.

    Args:
        size: The dimension of the square kernel to generate.
        sigma: The standard deviation of the Gaussian.

    Returns:
        An image object.
    """
    kernel = create_image_from_param(
        Gauss2DParam.create(height=size, width=size, sigma=sigma)
    )
    return kernel


def _generate_identity_kernel(size: int = 7) -> ImageObj:
    """Generate an identity kernel image.

    Args:
        size: The dimension of the square kernel to generate (must be odd).

    Returns:
        An image object.
    """
    assert size % 2 == 1, "Identity kernel size must be odd."
    kernel = create_image_from_param(Zero2DParam.create(height=size, width=size))
    assert kernel.data is not None
    kernel.data[size // 2, size // 2] = 1.0
    return kernel


def __convolve_image(kernel: ImageObj, size: int = 32) -> tuple[ImageObj, ImageObj]:
    """Generate a test image and its convolution with a Gaussian kernel.

    Returns:
        A tuple (source image, kernel, convolved image).
    """
    original = _generate_rectangle_image(size=size)
    assert original.data is not None
    convolved = convolution(original, kernel)
    assert convolved.data is not None
    return original, convolved


@pytest.mark.validation
def test_image_convolution(disable_kernel_normalization, size: int = 32) -> None:
    """Validation test for the image convolution processing.

    Note: This test disables kernel normalization to compare against raw scipy results.
    """
    # Test with a Gaussian kernel.
    kernel = _generate_gaussian_kernel(size=size, sigma=1.0)
    original, convolved = __convolve_image(kernel, size=size)
    exp = sps.convolve(original.data, kernel.data, mode="same", method="auto")
    check_array_result("Convolution", convolved.data, exp)
    guiutils.view_images_side_by_side_if_gui(
        [original, kernel, convolved],
        ["Original", "Kernel", "Convolved"],
        title="Image Convolution Test: Gaussian Kernel",
    )
    # Test with an identity kernel.
    kernel = _generate_identity_kernel(size=7)
    original, convolved = __convolve_image(kernel, size=size)
    # check_array_result("Convolution identity", convolved.data, original.data)
    guiutils.view_images_side_by_side_if_gui(
        [original, kernel, convolved],
        ["Original", "Kernel", "Convolved"],
        title="Image Convolution Test: Identity Kernel",
    )


@pytest.mark.validation
def test_image_deconvolution(disable_kernel_normalization, size: int = 32) -> None:
    """Validation test for image deconvolution.

    Note: This test disables kernel normalization to compare against expected results.
    """
    # Test with an identity kernel.
    kernel = _generate_identity_kernel(size=31)
    original, convolved = __convolve_image(kernel, size=size)
    deconvolved = deconvolution(convolved, kernel)
    guiutils.view_images_side_by_side_if_gui(
        [original, kernel, convolved, deconvolved],
        ["Original", "Kernel", "Convolved", "Deconvolved"],
        title="Image Deconvolution Test: Identity Kernel",
    )
    check_array_result("Deconvolution identity", deconvolved.data, original.data)

    # # Test with a Gaussian kernel.
    kernel = _generate_gaussian_kernel(size=31, sigma=1.0)
    original, convolved = __convolve_image(kernel, size=64)
    deconvolved = deconvolution(convolved, kernel)
    # check_array_result("Deconvolution Gaussian", deconvolved.data, original.data)
    guiutils.view_images_side_by_side_if_gui(
        [original, kernel, convolved, deconvolved],
        ["Original", "Kernel", "Convolved", "Deconvolved"],
        title="Image Deconvolution Test: Gaussian Kernel",
    )


def test_tools_image_deconvolve_null_kernel() -> None:
    """Test deconvolution with a null kernel."""
    size = 32
    src = _generate_image(size)
    assert src.data is not None
    kernel = create_image_from_param(Zero2DParam.create(height=size, width=size))
    assert kernel.data is not None
    with pytest.raises(ValueError, match="Deconvolution kernel cannot be null."):
        deconvolve(src.data, kernel.data)


if __name__ == "__main__":
    guiutils.enable_gui()
    test_image_convolution()
    test_image_deconvolution()
    test_tools_image_deconvolve_null_kernel()
