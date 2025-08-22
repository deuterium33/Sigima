# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for image noise addition."""

from __future__ import annotations

import numpy as np
import pytest

import sigima.objects
import sigima.proc.image
from sigima.objects.base import NormalDistributionParam
from sigima.tests import guiutils
from sigima.tests.helpers import check_array_result, check_scalar_result

test_add_gaussian_noise_parameters = [1, 2, 5]


@pytest.mark.parametrize("multiple", test_add_gaussian_noise_parameters)
@pytest.mark.validation
def test_add_gaussian_noise_image(multiple: int) -> None:
    """Test :py:func:`sigima.proc.image.add_gaussian_noise`."""
    # Create a clean test image (no added random noise).
    size = multiple * 512
    param = sigima.objects.Gauss2DParam.create(height=size, width=size)
    ima = sigima.objects.create_image_from_param(param)

    p = NormalDistributionParam.create(seed=42, mu=0.0, sigma=1.0)
    # Run twice with same parameters to check reproducibility.
    res1 = sigima.proc.image.add_gaussian_noise(ima, p)
    res2 = sigima.proc.image.add_gaussian_noise(ima, p)

    guiutils.view_images_side_by_side_if_gui_enabled(
        [ima, res1, res2], ["Clean", "Noisy", "Noisy"]
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


if __name__ == "__main__":
    guiutils.set_current_request(gui=True)
    for parameter in test_add_gaussian_noise_parameters:
        test_add_gaussian_noise_image(parameter)
