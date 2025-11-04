# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for image noise addition."""

from __future__ import annotations

import numpy as np
import pytest

import sigima.objects
import sigima.proc.image
from sigima.objects import (
    NormalDistribution2DParam,
    PoissonDistribution2DParam,
    UniformDistribution2DParam,
)
from sigima.tests import guiutils
from sigima.tests.helpers import check_array_result, check_scalar_result


@pytest.mark.validation
def test_image_add_gaussian_noise() -> None:
    """Test :py:func:`sigima.proc.image.add_gaussian_noise`."""
    # Generate a clean image.
    size = 512
    param = sigima.objects.Gauss2DParam.create(height=size, width=size)
    ima = sigima.objects.create_image_from_param(param)
    # Add Gaussian noise.
    # Run twice with same parameters to check reproducibility.
    p = NormalDistribution2DParam.create(seed=42, mu=0.0, sigma=1.0)
    res1 = sigima.proc.image.add_gaussian_noise(ima, p)
    res2 = sigima.proc.image.add_gaussian_noise(ima, p)

    guiutils.view_images_side_by_side_if_gui(
        [ima, res1, res2],
        ["Clean", "Noisy (1)", "Noisy (2)"],
        title=f"Gaussian Noise Addition ({size}x{size}): "
        f"Noisy images should be identical (same seed)",
    )

    # Shape must be preserved.
    assert ima.data is not None
    assert res1.data is not None
    assert res1.data.shape == ima.data.shape

    # Check noise statistics.
    noise = res1.data - ima.data
    mean_noise = float(np.mean(noise))
    assert p.mu is not None
    assert p.sigma is not None
    expected_error = 5.0 * p.sigma / np.sqrt(ima.data.size)
    check_scalar_result("Mean noise", mean_noise, p.mu, atol=expected_error)

    # Identical results for same seed and distribution parameters.
    assert res2.data is not None
    check_array_result("Reproducibility", res1.data, res2.data)


@pytest.mark.validation
def test_image_add_poisson_noise() -> None:
    """Test :py:func:`sigima.proc.image.add_poisson_noise`."""
    # Generate a clean image.
    size = 512
    param = sigima.objects.Gauss2DParam.create(height=size, width=size)
    ima = sigima.objects.create_image_from_param(param)
    # Add Poisson noise.
    # Run twice with same parameters to check reproducibility.
    p = PoissonDistribution2DParam.create(seed=42, lam=10.0)
    res1 = sigima.proc.image.add_poisson_noise(ima, p)
    res2 = sigima.proc.image.add_poisson_noise(ima, p)

    guiutils.view_images_side_by_side_if_gui(
        [ima, res1, res2],
        ["Clean", "Noisy (1)", "Noisy (2)"],
        title=f"Poisson Noise Addition ({size}x{size}): "
        f"Noisy images should be identical (same seed)",
    )

    # Shape must be preserved.
    assert ima.data is not None
    assert res1.data is not None
    assert res1.data.shape == ima.data.shape

    # Check noise statistics.
    noise = res1.data - ima.data
    mean_noise = float(np.mean(noise))
    assert p.lam is not None
    # For Poisson distribution, variance equals the mean (λ)
    expected_error = 5.0 * np.sqrt(p.lam) / np.sqrt(ima.data.size)
    check_scalar_result("Mean noise", mean_noise, p.lam, atol=expected_error)

    # Identical results for same seed and distribution parameters.
    assert res2.data is not None
    check_array_result("Reproducibility", res1.data, res2.data)


@pytest.mark.validation
def test_image_add_uniform_noise() -> None:
    """Test :py:func:`sigima.proc.image.add_uniform_noise`."""
    # Generate a clean image.
    size = 512
    param = sigima.objects.Gauss2DParam.create(height=size, width=size)
    ima = sigima.objects.create_image_from_param(param)
    # Add uniform noise.
    # Run twice with same parameters to check reproducibility.
    p = UniformDistribution2DParam.create(seed=42, vmin=-1.0, vmax=1.0)
    res1 = sigima.proc.image.add_uniform_noise(ima, p)
    res2 = sigima.proc.image.add_uniform_noise(ima, p)

    guiutils.view_images_side_by_side_if_gui(
        [ima, res1, res2],
        ["Clean", "Noisy (1)", "Noisy (2)"],
        title=f"Uniform Noise Addition ({size}x{size}): "
        f"Noisy images should be identical (same seed)",
    )

    # Shape must be preserved.
    assert ima.data is not None
    assert res1.data is not None
    assert res1.data.shape == ima.data.shape

    # Check noise statistics.
    noise = res1.data - ima.data
    mean_noise = float(np.mean(noise))
    assert p.vmin is not None
    assert p.vmax is not None
    expected_mean = (p.vmin + p.vmax) / 2.0
    # For uniform distribution, variance = (b-a)^2 / 12
    variance = ((p.vmax - p.vmin) ** 2) / 12.0
    expected_error = 5.0 * np.sqrt(variance) / np.sqrt(ima.data.size)
    check_scalar_result("Mean noise", mean_noise, expected_mean, atol=expected_error)

    # Identical results for same seed and distribution parameters.
    assert res2.data is not None
    check_array_result("Reproducibility", res1.data, res2.data)


if __name__ == "__main__":
    guiutils.enable_gui()
    test_image_add_gaussian_noise()
    test_image_add_poisson_noise()
    test_image_add_uniform_noise()
