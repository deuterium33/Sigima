# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for morphology computation functions.
"""

from __future__ import annotations

from typing import Callable

import pytest
from skimage import morphology

import sigima.objects
import sigima.params
import sigima.proc.image
from sigima.tests.data import get_test_image
from sigima.tests.helpers import check_array_result


def __generic_morphology_validation(
    func: Callable[
        [sigima.objects.ImageObj, sigima.params.MorphologyParam],
        sigima.objects.ImageObj,
    ],
) -> None:
    """Generic test for morphology methods."""
    # See [1] in sigima\tests\image\__init__.py for more details about the validation.
    src = get_test_image("flower.npy")
    p = sigima.params.MorphologyParam.create(radius=10)
    dst: sigima.objects.ImageObj = func(src, p)
    method = func.__name__
    exp = getattr(morphology, method)(src.data, footprint=morphology.disk(p.radius))
    check_array_result(f"{method.capitalize()}[radius={p.radius}]", dst.data, exp)


@pytest.mark.validation
def test_white_tophat() -> None:
    """Validation test for the image white top-hat processing."""
    __generic_morphology_validation(sigima.proc.image.white_tophat)


@pytest.mark.validation
def test_black_tophat() -> None:
    """Validation test for the image black top-hat processing."""
    __generic_morphology_validation(sigima.proc.image.black_tophat)


@pytest.mark.validation
def test_erosion() -> None:
    """Validation test for the image erosion processing."""
    __generic_morphology_validation(sigima.proc.image.erosion)


@pytest.mark.validation
def test_dilation() -> None:
    """Validation test for the image dilation processing."""
    __generic_morphology_validation(sigima.proc.image.dilation)


@pytest.mark.validation
def test_opening() -> None:
    """Validation test for the image opening processing."""
    __generic_morphology_validation(sigima.proc.image.opening)


@pytest.mark.validation
def test_closing() -> None:
    """Validation test for the image closing processing."""
    __generic_morphology_validation(sigima.proc.image.closing)


if __name__ == "__main__":
    test_white_tophat()
    test_black_tophat()
    test_erosion()
    test_dilation()
    test_opening()
    test_closing()
