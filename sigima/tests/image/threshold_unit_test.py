# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for thresholding computation functions.
"""

from __future__ import annotations

from typing import Callable

import pytest
from skimage import filters, util

import sigima.objects
import sigima.params
import sigima.proc.image
from sigima.tests.data import get_test_image
from sigima.tests.helpers import check_array_result


@pytest.mark.validation
def test_threshold() -> None:
    """Validation test for the image threshold processing."""
    src = get_test_image("flower.npy")
    p = sigima.params.ThresholdParam.create(value=100.0)
    dst = sigima.proc.image.threshold(src, p)
    exp = util.img_as_ubyte(src.data > p.value)
    check_array_result(f"Threshold[{p.value}]", dst.data, exp)


def __generic_threshold_validation(
    func: Callable[
        [sigima.objects.ImageObj, sigima.params.ThresholdParam], sigima.objects.ImageObj
    ],
) -> None:
    """Generic test for thresholding methods."""
    # See [1] in sigima\tests\image\__init__.py for more details about the validation.
    src = get_test_image("flower.npy")
    method = func.__name__.replace("threshold_", "")
    dst = sigima.proc.image.threshold(
        src, sigima.params.ThresholdParam.create(method=method)
    )
    exp = util.img_as_ubyte(
        src.data > getattr(filters, f"threshold_{method}")(src.data)
    )
    check_array_result(f"Threshold{method.capitalize()}", dst.data, exp)
    # Checking that we have the same result with direct function call
    dst2 = func(src)
    check_array_result(f"Threshold{method.capitalize()} (direct)", dst2.data, exp)


@pytest.mark.validation
def test_threshold_isodata() -> None:
    """Validation test for the image threshold Isodata processing."""
    __generic_threshold_validation(sigima.proc.image.threshold_isodata)


@pytest.mark.validation
def test_threshold_li() -> None:
    """Validation test for the image threshold Li processing."""
    __generic_threshold_validation(sigima.proc.image.threshold_li)


@pytest.mark.validation
def test_threshold_mean() -> None:
    """Validation test for the image threshold Mean processing."""
    __generic_threshold_validation(sigima.proc.image.threshold_mean)


@pytest.mark.validation
def test_threshold_minimum() -> None:
    """Validation test for the image threshold Minimum processing."""
    __generic_threshold_validation(sigima.proc.image.threshold_minimum)


@pytest.mark.validation
def test_threshold_otsu() -> None:
    """Validation test for the image threshold Otsu processing."""
    __generic_threshold_validation(sigima.proc.image.threshold_otsu)


@pytest.mark.validation
def test_threshold_triangle() -> None:
    """Validation test for the image threshold Triangle processing."""
    __generic_threshold_validation(sigima.proc.image.threshold_triangle)


@pytest.mark.validation
def test_threshold_yen() -> None:
    """Validation test for the image threshold Yen processing."""
    __generic_threshold_validation(sigima.proc.image.threshold_yen)


if __name__ == "__main__":
    test_threshold()
    test_threshold_isodata()
    test_threshold_li()
    test_threshold_mean()
    test_threshold_minimum()
    test_threshold_otsu()
    test_threshold_triangle()
    test_threshold_yen()
