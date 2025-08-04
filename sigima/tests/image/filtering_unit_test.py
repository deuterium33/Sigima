# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for image filtering functions.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.ndimage as spi
import scipy.signal as sps
from skimage import filters

import sigima.params
import sigima.proc.image
import sigima.tools.image
from sigima.objects import ImageObj
from sigima.objects.image import create_image
from sigima.tests import guiutils
from sigima.tests.data import get_test_image
from sigima.tests.helpers import check_array_result, check_scalar_result


@pytest.mark.validation
def test_image_gaussian_filter() -> None:
    """Validation test for the image Gaussian filter processing."""
    src = get_test_image("flower.npy")
    for sigma in (10.0, 50.0):
        p = sigima.params.GaussianParam.create(sigma=sigma)
        dst = sigima.proc.image.gaussian_filter(src, p)
        exp = spi.gaussian_filter(src.data, sigma=sigma)
        check_array_result(f"GaussianFilter[sigma={sigma}]", dst.data, exp)


@pytest.mark.validation
def test_image_moving_average() -> None:
    """Validation test for the image moving average processing."""
    src = get_test_image("flower.npy")
    p = sigima.params.MovingAverageParam.create(n=30)
    for mode in p.modes:
        p.mode = mode
        dst = sigima.proc.image.moving_average(src, p)
        exp = spi.uniform_filter(src.data, size=p.n, mode=p.mode)
        check_array_result(f"MovingAvg[n={p.n},mode={p.mode}]", dst.data, exp)


@pytest.mark.validation
def test_image_moving_median() -> None:
    """Validation test for the image moving median processing."""
    src = get_test_image("flower.npy")
    p = sigima.params.MovingMedianParam.create(n=5)
    for mode in p.modes:
        p.mode = mode
        dst = sigima.proc.image.moving_median(src, p)
        exp = spi.median_filter(src.data, size=p.n, mode=p.mode)
        check_array_result(f"MovingMed[n={p.n},mode={p.mode}]", dst.data, exp)


@pytest.mark.validation
def test_image_wiener() -> None:
    """Validation test for the image Wiener filter processing."""
    src = get_test_image("flower.npy")
    dst = sigima.proc.image.wiener(src)
    exp = sps.wiener(src.data)
    check_array_result("Wiener", dst.data, exp)


@pytest.mark.validation
def test_butterworth() -> None:
    """Validation test for the image Butterworth filter processing."""
    src = get_test_image("flower.npy")
    p = sigima.params.ButterworthParam.create(order=2, cut_off=0.5, high_pass=False)
    dst = sigima.proc.image.butterworth(src, p)
    exp = filters.butterworth(src.data, p.cut_off, p.high_pass, p.order)
    check_array_result(
        f"Butterworth[order={p.order},cut_off={p.cut_off},high_pass={p.high_pass}]",
        dst.data,
        exp,
    )


def build_clean_noisy_images(
    shape: tuple[int, int] = (64, 64), noise_level: float = 0.4, freq: float = 0.05
) -> tuple[ImageObj, ImageObj]:
    """Generate a test image with a low-frequency signal and high-frequency noise.

    Args:
        shape: Shape of the image (height, width).
        noise_level: Standard deviation of the Gaussian noise.
        freq: Frequency of the low-frequency signal.

    Returns:
        Tuple of (clean, noisy) where:
        - clean: The low-frequency component of the image.
        - noisy: The noisy image with added Gaussian noise.
    """
    # Low frequency signal: sinusoid with frequency in px^-1
    x = np.arange(shape[1])
    y = np.arange(shape[0])
    x_matrice, y_matrice = np.meshgrid(x, y)
    low_freq = 0.5 * (np.sin(2 * np.pi * freq * x_matrice)) + 0.5 * (
        np.sin(2 * np.pi * freq * y_matrice)
    )

    # High frequency noise
    rng = np.random.default_rng(seed=0)
    high_freq_noise = noise_level * rng.standard_normal(size=shape)

    # Sum of the two components
    img = low_freq + high_freq_noise
    return create_image("clean", low_freq), create_image("noisy", img)


@pytest.mark.validation
def test_freq_domain_gaussian_filter(request: pytest.FixtureRequest = None) -> None:
    """Validation test for :py:func:`sigima.tools.image.freq_domain_gaussian_filter`."""
    clean, noisy = build_clean_noisy_images(freq=0.05)
    param = sigima.proc.image.FreqDomainGaussianParam.create(f0=0.05, sigma=0.05)
    filt = sigima.proc.image.freq_domain_gaussian_filter(noisy, param)
    clean_area = clean.data[10:-10, 10:-10]

    guiutils.set_current_request(request)
    if guiutils.is_gui_enabled():
        # pylint: disable=import-outside-toplevel
        from guidata.qthelpers import qt_app_context

        from sigima.tests.vistools import view_images_side_by_side

        with qt_app_context():
            view_images_side_by_side(
                [clean, noisy, filt], titles=["Clean", "Noisy", "Filtered"]
            )

    mean_noise = float(np.mean(np.abs(clean_area - filt.data[10:-10, 10:-10])))
    check_scalar_result(
        "freq_domain_gaussian_filter noise reduction", mean_noise, 0.0, atol=0.1
    )


def test_freq_domain_gaussian_filter_constant_image() -> None:
    """Edge case: filtering a constant image must preserve the constant value
    (DC component)."""
    img_const = np.full((64, 64), fill_value=7.42)
    zout = sigima.tools.image.freq_domain_gaussian_filter(img_const, f0=0.0, sigma=0.05)
    # Ignore borders
    center = zout[10:-10, 10:-10]
    # Assert that all values are (almost) equal to the original constant
    assert np.allclose(center, 7.42, atol=1e-10), "Filtering constant image failed"


def test_freq_domain_gaussian_filter_symmetry() -> None:
    """Test: filtering a symmetric image yields a symmetric result."""
    # Create a symmetric image (e.g., a centered 2D Gaussian)
    x = np.linspace(-1, 1, 64)
    y = np.linspace(-1, 1, 64)
    xv, yv = np.meshgrid(x, y)
    img = np.exp(-(xv**2 + yv**2) / 0.1)

    zout = sigima.tools.image.freq_domain_gaussian_filter(img, f0=0.05, sigma=0.02)
    # Symmetry check: image must be (almost) symmetric along both axes
    assert np.allclose(zout, zout[::-1, :], atol=1e-10), "Vertical symmetry lost"
    assert np.allclose(zout, zout[:, ::-1], atol=1e-10), "Horizontal symmetry lost"


if __name__ == "__main__":
    test_image_gaussian_filter()
    test_image_moving_average()
    test_image_moving_median()
    test_image_wiener()
    test_butterworth()
    test_freq_domain_gaussian_filter(request=guiutils.DummyRequest(gui=True))
    test_freq_domain_gaussian_filter_constant_image()
    test_freq_domain_gaussian_filter_symmetry()
