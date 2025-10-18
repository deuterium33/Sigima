# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Signal stability analysis unit test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

import sigima.objects
import sigima.params
import sigima.proc.signal
from sigima.tests.helpers import check_array_result


def get_optimal_points(test_func: Callable) -> int:
    """Return optimal number of points for different algorithms.

    Args:
        test_func: Test function object to determine algorithm type

    Returns:
        Optimal number of points balancing accuracy and performance
    """
    func_name = test_func.__name__
    if "overlapping" in func_name or "hadamard" in func_name:
        return 500  # Minimum for meaningful results
    if "modified" in func_name or "total" in func_name:
        return 1000  # Need more for averaging
    # allan_variance, allan_deviation, time_deviation
    return 2000  # Balance between speed and accuracy


def generate_white_noise(n_points: int, sigma=1.0) -> np.ndarray:
    """Generate white noise with known characteristics.

    Args:
        n_points: Number of data points
        sigma: Standard deviation of the white noise (default is 1.0)

    Returns:
        Array of white noise values
    """
    return np.random.normal(0, sigma, n_points)


def theoretical_allan_variance_white_noise(tau: float, sigma: float) -> float:
    """Calculate theoretical Allan variance for white noise.

    For white noise: AVAR(τ) = σ²/(2τ)
    But the Allan variance is computed as AVAR(τ) = σ²τ/τ = σ²τ because of the
    overlapping nature of the samples.

    Args:
        tau: Averaging time
        sigma: Standard deviation of the white noise

    Returns:
        Allan variance value for the given tau and sigma
    """
    return sigma**2 / tau


def generate_drift_signal(
    n_points: int, slope: float, intercept: float = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a linear drift signal.

    Args:
        n_points: Number of data points
        slope: Slope of the linear drift
        intercept: Intercept of the linear drift (default is 0)

    Returns:
        A tuple of (time array, values array)
    """
    time = np.arange(n_points)
    values = slope * time + intercept
    return time, values


def theoretical_allan_variance_drift(tau: float, slope: float) -> float:
    """Theoretical Allan variance for a drift signal.

    Args:
        tau: Averaging time
        slope: Slope of the linear drift

    Returns:
        Allan variance value for the given tau and slope
    """
    return (slope**2 * tau**2) / 2


@pytest.mark.validation
def test_signal_allan_variance():
    """Test Allan variance computation against theoretical values."""
    n_points = get_optimal_points(test_signal_allan_variance)
    sigma = 1.0

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate and test white noise signal
    time_white = np.arange(n_points)
    values_white = generate_white_noise(n_points, sigma)
    sig1 = sigima.objects.create_signal("White Noise Test", time_white, values_white)

    # Define Allan variance parameters - limit max_tau for more reliable statistics
    param = sigima.params.AllanVarianceParam()
    param.max_tau = 20  # Limited to ensure sufficient samples for averaging

    # Compute Allan variance using the high-level function
    res1 = sigima.proc.signal.allan_variance(sig1, param)
    th_av_white = theoretical_allan_variance_white_noise(res1.x, sigma)

    # Use relative tolerance for white noise (statistical variation scales)
    # 20% tolerance accounts for statistical variance in Allan estimator
    check_array_result(
        "White noise Allan variance",
        res1.y,
        th_av_white,
        rtol=0.20,
        atol=0.005,
    )

    # Generate and test drift signal (deterministic - same seed not needed)
    slope = 0.01
    time, values = generate_drift_signal(n_points, slope)
    sig2 = sigima.objects.create_signal("Drift Test", time, values)

    # Compute Allan variance using the high-level function
    res2 = sigima.proc.signal.allan_variance(sig2, param)
    th_av_drift = theoretical_allan_variance_drift(res2.x, slope)

    # Drift is deterministic, tighter tolerances apply
    check_array_result(
        "Drift Allan variance",
        res2.y,
        th_av_drift,
        rtol=0.05,
        atol=0.0001,
    )


@pytest.mark.validation
def test_signal_allan_deviation():
    """Test Allan deviation computation against theoretical values."""
    n_points = get_optimal_points(test_signal_allan_deviation)
    sigma = 1.0

    # Set random seed for reproducibility
    np.random.seed(43)

    # Generate and test white noise signal
    time_white = np.arange(n_points)
    values_white = generate_white_noise(n_points, sigma)
    sig1 = sigima.objects.create_signal("White Noise Test", time_white, values_white)

    # Define Allan variance parameters - limit max_tau for more reliable statistics
    param = sigima.params.AllanVarianceParam()
    param.max_tau = 20  # Limited to ensure sufficient samples for averaging

    # Compute Allan deviation using the high-level function
    res1 = sigima.proc.signal.allan_deviation(sig1, param)
    th_av_white = theoretical_allan_variance_white_noise(res1.x, sigma)

    # Use relative tolerance for white noise (statistical variation scales)
    # 20% tolerance accounts for statistical variance in Allan estimator
    check_array_result(
        "White noise Allan deviation",
        res1.y,
        np.sqrt(th_av_white),
        rtol=0.20,
        atol=0.005,
    )

    # Generate and test drift signal (deterministic - same seed not needed)
    slope = 0.01
    time, values = generate_drift_signal(n_points, slope)
    sig2 = sigima.objects.create_signal("Drift Test", time, values)

    # Compute Allan deviation using the high-level function
    res2 = sigima.proc.signal.allan_deviation(sig2, param)
    th_av_drift = theoretical_allan_variance_drift(res2.x, slope)

    # Drift is deterministic, tighter tolerances apply
    check_array_result(
        "Drift Allan deviation",
        res2.y,
        np.sqrt(th_av_drift),
        rtol=0.05,
        atol=0.0001,
    )


@pytest.mark.validation
def test_signal_overlapping_allan_variance():
    """Test Overlapping Allan variance computation."""
    n_points = get_optimal_points(test_signal_overlapping_allan_variance)
    sigma = 1.0

    # Generate and test white noise signal
    time_white = np.arange(n_points)
    values_white = generate_white_noise(n_points, sigma)
    sig1 = sigima.objects.create_signal("White Noise Test", time_white, values_white)

    # Define Allan variance parameters
    param = sigima.params.AllanVarianceParam()
    param.max_tau = 50

    # Compute Overlapping Allan variance using the high-level function
    res1 = sigima.proc.signal.overlapping_allan_variance(sig1, param)

    # Overlapping Allan variance should produce finite, positive results
    assert len(res1.y) > 0, "Overlapping Allan variance should produce results"
    valid_values = res1.y[~np.isnan(res1.y)]
    assert len(valid_values) > 0, "Overlapping Allan variance should have valid values"
    assert np.all(valid_values > 0), "Overlapping Allan variance should be positive"

    # For white noise, overlapping Allan variance should be related to the noise level
    # and generally decrease with tau (though not necessarily following exact 1/tau)
    expected_order = sigma**2
    assert np.mean(valid_values) < 10 * expected_order, (
        "Overlapping Allan variance should be reasonable for white noise"
    )

    # Generate and test drift signal
    slope = 0.01
    time, values = generate_drift_signal(n_points, slope)
    sig2 = sigima.objects.create_signal("Drift Test", time, values)

    # Compute Overlapping Allan variance using the high-level function
    res2 = sigima.proc.signal.overlapping_allan_variance(sig2, param)

    # For drift signals, overlapping Allan variance should also be finite and positive
    valid_drift_values = res2.y[~np.isnan(res2.y)]
    assert len(valid_drift_values) > 0, (
        "Overlapping Allan variance should work for drift"
    )
    assert np.all(valid_drift_values > 0), (
        "Overlapping Allan variance should be positive"
    )

    # Compare with regular Allan variance to ensure overlapping version
    # produces reasonable results
    res_regular = sigima.proc.signal.allan_variance(sig1, param)
    valid_regular = res_regular.y[~np.isnan(res_regular.y)]
    if len(valid_regular) > 0 and len(valid_values) > 0:
        # Results should be of similar order of magnitude
        ratio = np.mean(valid_values) / np.mean(valid_regular)
        assert 0.1 < ratio < 10, (
            "Overlapping and regular Allan variance should be of similar magnitude"
        )


@pytest.mark.validation
def test_signal_modified_allan_variance():
    """Test Modified Allan variance computation."""
    n_points = get_optimal_points(test_signal_modified_allan_variance)
    sigma = 1.0

    # Generate and test white noise signal
    time_white = np.arange(n_points)
    values_white = generate_white_noise(n_points, sigma)
    sig1 = sigima.objects.create_signal("White Noise Test", time_white, values_white)

    # Define Allan variance parameters
    param = sigima.params.AllanVarianceParam()
    param.max_tau = 50

    # Compute Modified Allan variance using the high-level function
    res1 = sigima.proc.signal.modified_allan_variance(sig1, param)

    # For white noise, Modified Allan variance should be proportional to 1/tau
    # The exact relationship depends on the specific implementation
    # We check that values are reasonable and decrease with tau
    assert len(res1.y) > 0, "Modified Allan variance should produce results"
    assert np.all(res1.y[~np.isnan(res1.y)] > 0), (
        "Modified Allan variance should be positive"
    )

    # For white noise, Modified Allan variance typically decreases with tau
    valid_indices = ~np.isnan(res1.y)
    if np.sum(valid_indices) > 1:
        valid_y = res1.y[valid_indices]
        # Check general decreasing trend for first few points
        if len(valid_y) >= 3:
            assert valid_y[2] < valid_y[0], (
                "Modified Allan variance should generally decrease for white noise"
            )


@pytest.mark.validation
def test_signal_hadamard_variance():
    """Test Hadamard variance computation."""
    n_points = get_optimal_points(test_signal_hadamard_variance)
    sigma = 1.0

    # Generate and test white noise signal
    time_white = np.arange(n_points)
    values_white = generate_white_noise(n_points, sigma)
    sig1 = sigima.objects.create_signal("White Noise Test", time_white, values_white)

    # Define Allan variance parameters
    param = sigima.params.AllanVarianceParam()
    param.max_tau = 50

    # Compute Hadamard variance using the high-level function
    res1 = sigima.proc.signal.hadamard_variance(sig1, param)

    # For white noise, Hadamard variance should be finite and positive
    assert len(res1.y) > 0, "Hadamard variance should produce results"
    valid_values = res1.y[~np.isnan(res1.y)]
    assert len(valid_values) > 0, "Hadamard variance should have valid values"
    assert np.all(valid_values > 0), "Hadamard variance should be positive"

    # Generate and test linear drift signal
    # (Hadamard variance is robust to linear drift)
    slope = 0.01
    time, values = generate_drift_signal(n_points, slope)
    sig2 = sigima.objects.create_signal("Drift Test", time, values)

    # Compute Hadamard variance for drift signal
    res2 = sigima.proc.signal.hadamard_variance(sig2, param)

    # Hadamard variance should be less sensitive to linear drift than Allan variance
    # For pure linear drift, Hadamard variance should be close to zero or very small
    valid_drift_values = res2.y[~np.isnan(res2.y)]
    if len(valid_drift_values) > 0:
        # Hadamard variance should be smaller for drift signals
        assert np.mean(valid_drift_values) < np.mean(valid_values), (
            "Hadamard variance should be smaller for drift signals than white noise"
        )


@pytest.mark.validation
def test_signal_total_variance():
    """Test Total variance computation."""
    n_points = get_optimal_points(test_signal_total_variance)
    sigma = 1.0

    # Generate and test white noise signal
    time_white = np.arange(n_points)
    values_white = generate_white_noise(n_points, sigma)
    sig1 = sigima.objects.create_signal("White Noise Test", time_white, values_white)

    # Define Allan variance parameters
    param = sigima.params.AllanVarianceParam()
    param.max_tau = 50

    # Compute Total variance using the high-level function
    res1 = sigima.proc.signal.total_variance(sig1, param)

    # Total variance should be finite and positive
    assert len(res1.y) > 0, "Total variance should produce results"
    valid_values = res1.y[~np.isnan(res1.y)]
    assert len(valid_values) > 0, "Total variance should have valid values"
    assert np.all(valid_values > 0), "Total variance should be positive"

    # For white noise, total variance should be related to the noise level
    # and should be of the same order of magnitude as the square of the noise
    expected_order = sigma**2
    assert np.mean(valid_values) < 100 * expected_order, (
        "Total variance should be reasonable for white noise"
    )


@pytest.mark.validation
def test_signal_time_deviation():
    """Test Time Deviation computation."""
    n_points = get_optimal_points(test_signal_time_deviation)
    sigma = 1.0

    # Generate and test white noise signal
    time_white = np.arange(n_points)
    values_white = generate_white_noise(n_points, sigma)
    sig1 = sigima.objects.create_signal("White Noise Test", time_white, values_white)

    # Define Allan variance parameters
    param = sigima.params.AllanVarianceParam()
    param.max_tau = 50

    # Compute Time Deviation using the high-level function
    res1 = sigima.proc.signal.time_deviation(sig1, param)

    # Time deviation should be finite and positive
    assert len(res1.y) > 0, "Time deviation should produce results"
    valid_values = res1.y[~np.isnan(res1.y)]
    assert len(valid_values) > 0, "Time deviation should have valid values"
    assert np.all(valid_values > 0), "Time deviation should be positive"

    # Time deviation is related to Allan variance: TDEV = sqrt(AVAR) * tau
    # So it should increase with tau for white noise
    if len(valid_values) >= 2:
        valid_x = res1.x[~np.isnan(res1.y)]
        # Check that time deviation generally increases with tau for white noise
        correlation = np.corrcoef(valid_x[: len(valid_values)], valid_values)[0, 1]
        assert correlation > 0.5, (
            "Time deviation should generally increase with tau for white noise"
        )

    # Compare with Allan deviation to verify the relationship
    res_adev = sigima.proc.signal.allan_deviation(sig1, param)

    # TDEV = ADEV * tau (approximately)
    # Check this relationship for some values
    for i, tau in enumerate(res1.x[: min(5, len(res1.x))]):
        if not np.isnan(res1.y[i]) and not np.isnan(res_adev.y[i]) and tau > 0:
            expected_tdev = res_adev.y[i] * tau
            relative_error = abs(res1.y[i] - expected_tdev) / expected_tdev
            assert relative_error < 0.1, (
                f"Time deviation should match ADEV * tau relationship at tau={tau}"
            )


if __name__ == "__main__":
    test_signal_allan_variance()
    test_signal_allan_deviation()
    test_signal_overlapping_allan_variance()
    test_signal_modified_allan_variance()
    test_signal_hadamard_variance()
    test_signal_total_variance()
    test_signal_time_deviation()
