# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for image convolution/deconvolution features."""

from __future__ import annotations

import pytest
import scipy.signal as sps

from sigima.objects import create_image_from_param
from sigima.objects.image import Gauss2DParam, ImageObj, Zeros2DParam
from sigima.proc.image.mathops import convolution, deconvolution
from sigima.tests import guiutils
from sigima.tests.helpers import check_array_result
from sigima.tools.image import deconvolve


def _generate_image(size: int = 16) -> ImageObj:
    """Generate a test square image.

    Args:
        size: The dimension of the square image to generate.

    Returns:
        An image object.
    """
    # Gaussian image.
    gauss_img = create_image_from_param(Gauss2DParam.create(height=size, width=size))
    return gauss_img


@pytest.mark.validation
def test_image_convolution() -> None:
    """Validation test for the image convolution processing."""
    size = 32
    src1 = _generate_image(size)
    assert src1.data is not None
    src2 = create_image_from_param(
        Gauss2DParam.create(height=size, width=size, sigma=10.0)
    )
    assert src2.data is not None
    dst = convolution(src1, src2)
    assert dst.data is not None
    exp = sps.convolve(src1.data, src2.data, mode="same", method="auto")
    check_array_result("Convolution", dst.data, exp)


@pytest.mark.validation
def test_image_deconvolution() -> None:
    """Validation test for image deconvolution."""
    size = 32
    src = _generate_image(size)
    assert src.data is not None
    # Identity kernel.
    kernel = create_image_from_param(Zeros2DParam.create(height=size, width=size))
    assert kernel.data is not None
    kernel.data[0, 0] = 1.0
    # Deconvolve the image.
    result = deconvolution(src, kernel)
    assert result.data is not None
    # View the images.
    guiutils.view_images_side_by_side_if_gui(
        [src, kernel, result],
        ["Original", "Kernel", "Deconvolved"],
    )
    # The deconvolution should be identical to the source.
    check_array_result("Computation image deconvolve test", result.data, src.data)


def test_tools_image_deconvolve_null_kernel() -> None:
    """Test deconvolution with a null kernel."""
    size = 32
    src = _generate_image(size)
    assert src.data is not None
    kernel = create_image_from_param(Zeros2DParam.create(height=size, width=size))
    assert kernel.data is not None
    with pytest.raises(ValueError, match="Deconvolution kernel cannot be null."):
        deconvolve(src.data, kernel.data)


if __name__ == "__main__":
    guiutils.enable_gui()
    test_image_deconvolution()
    test_tools_image_deconvolve_null_kernel()
