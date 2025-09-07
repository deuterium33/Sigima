# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for edge detection computation functions.
"""

from __future__ import annotations

import pytest
from skimage import feature, filters, util

import sigima.objects
import sigima.params
import sigima.proc.image
from sigima.tests.data import get_test_image
from sigima.tests.helpers import check_array_result


@pytest.mark.validation
def test_canny() -> None:
    """Validation test for the image Canny edge detection processing."""
    # See [1] in sigima\tests\image\__init__.py for more details about the validation.
    src = get_test_image("flower.npy")
    p = sigima.params.CannyParam.create(
        sigma=1.0, low_threshold=0.1, high_threshold=0.2
    )
    dst = sigima.proc.image.canny(src, p)
    exp = util.img_as_ubyte(
        feature.canny(
            src.data,
            sigma=p.sigma,
            low_threshold=p.low_threshold,
            high_threshold=p.high_threshold,
            use_quantiles=p.use_quantiles,
            mode=p.mode.value,
            cval=p.cval,
        )
    )
    check_array_result(
        f"Canny[sigma={p.sigma},low_threshold={p.low_threshold},"
        f"high_threshold={p.high_threshold}]",
        dst.data,
        exp,
    )


def __generic_edge_validation(method: str) -> None:
    """Generic test for edge detection methods."""
    # See [1] in sigima\tests\image\__init__.py for more details about the validation.
    src = get_test_image("flower.npy")
    dst: sigima.objects.ImageObj = getattr(sigima.proc.image, method)(src)
    exp = getattr(filters, method)(src.data)
    check_array_result(f"{method.capitalize()}", dst.data, exp)


@pytest.mark.validation
def test_roberts() -> None:
    """Validation test for the image Roberts edge detection processing."""
    __generic_edge_validation("roberts")


@pytest.mark.validation
def test_prewitt() -> None:
    """Validation test for the image Prewitt edge detection processing."""
    __generic_edge_validation("prewitt")


@pytest.mark.validation
def test_prewitt_h() -> None:
    """Validation test for the image horizontal Prewitt edge detection processing."""
    __generic_edge_validation("prewitt_h")


@pytest.mark.validation
def test_prewitt_v() -> None:
    """Validation test for the image vertical Prewitt edge detection processing."""
    __generic_edge_validation("prewitt_v")


@pytest.mark.validation
def test_sobel() -> None:
    """Validation test for the image Sobel edge detection processing."""
    __generic_edge_validation("sobel")


@pytest.mark.validation
def test_sobel_h() -> None:
    """Validation test for the image horizontal Sobel edge detection processing."""
    __generic_edge_validation("sobel_h")


@pytest.mark.validation
def test_sobel_v() -> None:
    """Validation test for the image vertical Sobel edge detection processing."""
    __generic_edge_validation("sobel_v")


@pytest.mark.validation
def test_scharr() -> None:
    """Validation test for the image Scharr edge detection processing."""
    __generic_edge_validation("scharr")


@pytest.mark.validation
def test_scharr_h() -> None:
    """Validation test for the image horizontal Scharr edge detection processing."""
    __generic_edge_validation("scharr_h")


@pytest.mark.validation
def test_scharr_v() -> None:
    """Validation test for the image vertical Scharr edge detection processing."""
    __generic_edge_validation("scharr_v")


@pytest.mark.validation
def test_farid() -> None:
    """Validation test for the image Farid edge detection processing."""
    __generic_edge_validation("farid")


@pytest.mark.validation
def test_farid_h() -> None:
    """Validation test for the image horizontal Farid edge detection processing."""
    __generic_edge_validation("farid_h")


@pytest.mark.validation
def test_farid_v() -> None:
    """Validation test for the image vertical Farid edge detection processing."""
    __generic_edge_validation("farid_v")


@pytest.mark.validation
def test_laplace() -> None:
    """Validation test for the image Laplace edge detection processing."""
    __generic_edge_validation("laplace")


if __name__ == "__main__":
    test_canny()
    test_roberts()
    test_prewitt()
    test_prewitt_h()
    test_prewitt_v()
    test_sobel()
    test_sobel_h()
    test_sobel_v()
    test_scharr()
    test_scharr_h()
    test_scharr_v()
    test_farid()
    test_farid_h()
    test_farid_v()
    test_laplace()
