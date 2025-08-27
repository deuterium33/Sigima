# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for signal noise addition."""

from __future__ import annotations

import numpy as np
import pytest

import sigima.proc.signal
from sigima.objects import SinusParam, create_signal_from_param
from sigima.objects.signal import NormalDistribution1DParam
from sigima.tests import guiutils
from sigima.tests.helpers import check_array_result, check_scalar_result

test_add_gaussian_noise_parameters = [1, 10, 100]


@pytest.mark.parametrize("multiple", test_add_gaussian_noise_parameters)
@pytest.mark.validation
def test_add_gaussian_noise(multiple: int) -> None:
    """Test :py:func:`sigima.proc.signal.add_gaussian_noise`."""
    # Generate source signal.
    size = multiple * 1024
    param = SinusParam.create(size=size, freq=1.0)
    src = create_signal_from_param(param)

    p = NormalDistribution1DParam.create(seed=42, mu=0.0, sigma=0.1)
    # Run twice with same parameters to check reproducibility.
    res1 = sigima.proc.signal.add_gaussian_noise(src, p)
    res2 = sigima.proc.signal.add_gaussian_noise(src, p)

    guiutils.view_curves_if_gui([src, res1, res2])

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


if __name__ == "__main__":
    guiutils.enable_gui()
    for parameter in test_add_gaussian_noise_parameters:
        test_add_gaussian_noise(parameter)
