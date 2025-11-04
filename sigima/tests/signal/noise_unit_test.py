# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for signal noise addition."""

from __future__ import annotations

import numpy as np
import pytest

import sigima.proc.signal
from sigima.objects import SineParam, create_signal_from_param
from sigima.objects.signal import (
    NormalDistribution1DParam,
    PoissonDistribution1DParam,
    UniformDistribution1DParam,
)
from sigima.tests import guiutils
from sigima.tests.helpers import check_array_result, check_scalar_result


@pytest.mark.validation
def test_signal_add_gaussian_noise() -> None:
    """Test :py:func:`sigima.proc.signal.add_gaussian_noise`."""
    # Generate source signal.
    size = 1024
    param = SineParam.create(size=size, freq=1.0)
    src = create_signal_from_param(param)
    # Add Gaussian noise.
    # Run twice with same parameters to check reproducibility.
    p = NormalDistribution1DParam.create(seed=42, mu=0.0, sigma=0.1)
    res1 = sigima.proc.signal.add_gaussian_noise(src, p)
    res2 = sigima.proc.signal.add_gaussian_noise(src, p)

    guiutils.view_curves_if_gui(
        [src, res1, res2],
        title="Gaussian Noise Addition: Noisy images should be identical (same seed)",
    )

    # X-axis must be preserved.
    check_array_result("res1.x", res1.x, src.x)

    # Check noise statistics.
    noise = res1.y - src.y
    mean_noise = float(np.mean(noise))
    assert p.mu is not None
    assert p.sigma is not None
    expected_error = 5.0 * p.sigma / np.sqrt(src.x.size)
    check_scalar_result("Mean noise", mean_noise, p.mu, atol=expected_error)

    # Identical results for same seed and distribution parameters.
    check_array_result("Reproducibility", res1.y, res2.y)


@pytest.mark.validation
def test_signal_add_poisson_noise() -> None:
    """Test :py:func:`sigima.proc.signal.add_poisson_noise`."""
    # Generate source signal.
    size = 1024
    param = SineParam.create(size=size, freq=1.0)
    src = create_signal_from_param(param)
    # Add Poisson noise.
    # Run twice with same parameters to check reproducibility.
    p = PoissonDistribution1DParam.create(seed=42, lam=2.0)
    res1 = sigima.proc.signal.add_poisson_noise(src, p)
    res2 = sigima.proc.signal.add_poisson_noise(src, p)

    guiutils.view_curves_if_gui(
        [src, res1, res2],
        title="Poisson Noise Addition: Noisy signals should be identical (same seed)",
    )

    # X-axis must be preserved.
    check_array_result("res1.x", res1.x, src.x)

    # Check noise statistics.
    noise = res1.y - src.y
    mean_noise = float(np.mean(noise))
    assert p.lam is not None
    # For Poisson distribution, mean equals lambda, but we check with tolerance
    # since we're adding noise to an existing signal
    expected_error = 5.0 * np.sqrt(p.lam) / np.sqrt(src.x.size)
    check_scalar_result("Mean noise", mean_noise, p.lam, atol=expected_error)

    # Identical results for same seed and distribution parameters.
    check_array_result("Reproducibility", res1.y, res2.y)


@pytest.mark.validation
def test_signal_add_uniform_noise() -> None:
    """Test :py:func:`sigima.proc.signal.add_uniform_noise`."""
    # Generate source signal.
    size = 1024
    param = SineParam.create(size=size, freq=1.0)
    src = create_signal_from_param(param)
    # Add uniform noise.
    # Run twice with same parameters to check reproducibility.
    p = UniformDistribution1DParam.create(seed=42, vmin=-0.5, vmax=0.5)
    res1 = sigima.proc.signal.add_uniform_noise(src, p)
    res2 = sigima.proc.signal.add_uniform_noise(src, p)

    guiutils.view_curves_if_gui(
        [src, res1, res2],
        title="Uniform Noise Addition: Noisy signals should be identical (same seed)",
    )

    # X-axis must be preserved.
    check_array_result("res1.x", res1.x, src.x)

    # Check noise statistics.
    noise = res1.y - src.y
    mean_noise = float(np.mean(noise))
    assert p.vmin is not None
    assert p.vmax is not None
    expected_mean = (p.vmin + p.vmax) / 2.0
    # For uniform distribution, standard deviation is (vmax - vmin) / sqrt(12)
    expected_std = (p.vmax - p.vmin) / np.sqrt(12.0)
    expected_error = 5.0 * expected_std / np.sqrt(src.x.size)
    check_scalar_result("Mean noise", mean_noise, expected_mean, atol=expected_error)

    # Identical results for same seed and distribution parameters.
    check_array_result("Reproducibility", res1.y, res2.y)


if __name__ == "__main__":
    guiutils.enable_gui()
    test_signal_add_gaussian_noise()
    test_signal_add_poisson_noise()
    test_signal_add_uniform_noise()
