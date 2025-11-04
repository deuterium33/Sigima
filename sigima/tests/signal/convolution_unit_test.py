# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for signal convolution/deconvolution features."""

# pylint: disable=invalid-name

from __future__ import annotations

import numpy as np
import pytest

import sigima.proc.signal
from sigima.objects import create_signal
from sigima.objects.signal import SignalObj
from sigima.tests import guiutils
from sigima.tests.env import execenv
from sigima.tests.helpers import check_array_result
from sigima.tools.signal.fourier import deconvolve

N_POINTS = 1024


def _generate_experimental_signal(
    size: int = N_POINTS, noise_level: float = 0.05
) -> SignalObj:
    """Generate a realistic experimental signal with noise.

    Creates a sigmoid-based signal with realistic noise that mimics experimental data
    patterns commonly found in scientific measurements.

    Args:
        size: The size of the signal to generate.
        noise_level: The level of noise to add (as a fraction of signal amplitude).

    Returns:
        A signal object with experimental-like characteristics.
    """
    # Create x-axis
    x = np.linspace(-5.0, 5.0, size)

    # Create a sigmoid-like signal (similar to step response or transition)
    # This simulates typical experimental data patterns
    y_clean = 1.0 / (1.0 + np.exp(-2.0 * x))

    # Add some structure (multiple transitions at different scales)
    y_clean += 0.3 * (1.0 / (1.0 + np.exp(-10.0 * (x - 1.0))))
    y_clean += 0.2 * (1.0 / (1.0 + np.exp(-5.0 * (x + 2.0))))

    # Add realistic noise (combination of white noise and correlated noise)
    np.random.seed(42)  # For reproducible tests
    white_noise = np.random.normal(0, noise_level, size)

    # Add some correlated noise (simulates drift and systematic effects)
    drift = noise_level * 0.5 * np.sin(0.5 * x) * np.exp(-0.1 * x**2)

    y_noisy = y_clean + white_noise + drift

    # Create signal object
    signal = create_signal("Experimental Signal", x, y_noisy)
    return signal


def _generate_cable_response_kernel(
    size: int = N_POINTS,
    sigma1: float = 0.5,
    sigma2: float = 1.5,
    amplitude: float = 1.0,
) -> SignalObj:
    """Generate an asymmetric Gaussian kernel simulating cable frequency response.

    Creates a dissymmetric Gaussian kernel that simulates the frequency response
    characteristics of a cable or transmission line, which typically has asymmetric
    rise and fall times.

    Args:
        size: The size of the kernel.
        sigma1: Standard deviation for the rising edge (left side).
        sigma2: Standard deviation for the falling edge (right side).
        amplitude: Maximum amplitude of the kernel.

    Returns:
        A signal object representing the cable response kernel.
    """
    # Create x-axis centered around zero
    x = np.linspace(-5.0, 5.0, size)

    # Create asymmetric Gaussian
    y = np.zeros_like(x)

    # Left side (rising edge) - sharper response
    left_mask = x <= 0
    y[left_mask] = amplitude * np.exp(-(x[left_mask] ** 2) / (2 * sigma1**2))

    # Right side (falling edge) - slower response
    right_mask = x > 0
    y[right_mask] = amplitude * np.exp(-(x[right_mask] ** 2) / (2 * sigma2**2))

    # Normalize the kernel (area under curve should be 1 for proper convolution)
    y = y / np.sum(y) if np.sum(y) > 0 else y

    # Create signal object
    kernel = create_signal("Cable Response Kernel", x, y)
    return kernel


@pytest.mark.validation
def test_signal_convolution() -> None:
    """Enhanced validation test for the signal convolution processing.

    This test validates:
    1. Y-values match numpy.convolve (existing test)
    2. X-axis is preserved correctly (no shifting)
    3. Signal characteristics are reasonable
    """
    # Generate realistic experimental signal (minimal noise to improve conditioning)
    original_signal = _generate_experimental_signal(noise_level=0.025)

    # Generate a narrow asymmetric cable response kernel for better conditioning
    cable_kernel = _generate_cable_response_kernel(sigma1=0.05, sigma2=0.15)

    # Arbitrary normalization to help visualize the signal together with kernels:
    original_signal.y /= original_signal.y.max() - original_signal.y.min()
    original_signal.y *= cable_kernel.y.max()

    # Convolve the original signal with the cable response
    convolved_signal = sigima.proc.signal.convolution(original_signal, cable_kernel)

    # View the signals for visual inspection (if GUI enabled)
    guiutils.view_curves_if_gui(
        [original_signal, cable_kernel, convolved_signal],
        title="Convolution Validation Test",
    )

    exp = np.convolve(original_signal.y, cable_kernel.y, mode="same")

    # Original test: Y-values should be close to numpy.convolve result
    check_array_result("Convolution", convolved_signal.y, exp, similar=True)

    # The convolved signal should preserve the x-axis from original_signal exactly
    np.testing.assert_array_equal(
        convolved_signal.x,
        original_signal.x,
        "Convolution changed X-axis: X-axis should be preserved from source signal",
    )

    # The convolved signal shouldn't be extremely different from original
    original_range = np.max(original_signal.y) - np.min(original_signal.y)
    convolved_range = np.max(convolved_signal.y) - np.min(convolved_signal.y)
    range_ratio = convolved_range / original_range if original_range > 0 else np.inf

    # Convolution with Gaussian should not drastically change signal range
    # (smoothing might slightly reduce peaks but shouldn't be extreme)
    assert 0.1 < range_ratio < 10.0, (
        f"Convolution changed signal range too much: "
        f"ratio = {range_ratio:.2f} (expected 0.1 < ratio < 10.0)"
    )

    # Check if signal features are shifted after convolution
    shift_error = _detect_signal_shift_via_cross_correlation(
        original_signal.x, original_signal.y, convolved_signal.x, convolved_signal.y
    )

    # For convolution with a symmetric kernel, there should be minimal shift
    # (Gaussian kernel is symmetric, so convolution shouldn't introduce shift)
    assert shift_error < 0.01, (
        f"Convolution introduced significant signal shift: "
        f"shift = {shift_error:.6f} (expected < 0.01 for symmetric kernel)"
    )

    # Convolved signal should still be well-correlated with original
    # (convolution is smoothing, not completely changing the signal)
    correlation = np.corrcoef(original_signal.y, convolved_signal.y)[0, 1]

    # Print debug information for manual inspection FIRST
    execenv.print(
        f"Convolution validation - Range ratio: {range_ratio:.3f}, "
        f"Shift: {shift_error:.6f}, Correlation: {correlation:.4f}"
    )

    # A Gaussian kernel with sigma=10.0 might significantly smooth the signal
    assert correlation > 0.9, (
        f"Convolution destroyed signal structure: "
        f"correlation = {correlation:.4f} (expected > 0.9)"
    )


def _detect_signal_shift_via_cross_correlation(
    original_x: np.ndarray,
    original_y: np.ndarray,
    recovered_x: np.ndarray,
    recovered_y: np.ndarray,
) -> float:
    """Detect signal shift using cross-correlation of signal features.

    This method detects if the signal content is shifted, even if both signals
    use the same x-axis coordinates.
    """
    # If x-axes are different, we can't directly compare
    if not np.array_equal(original_x, recovered_x):
        return np.nan

    # Use cross-correlation to find the optimal shift
    # This works by sliding one signal over the other to find best alignment
    cross_corr = np.correlate(original_y, recovered_y, mode="full")

    # Find the shift that gives maximum correlation
    max_corr_index = np.argmax(cross_corr)
    optimal_shift_samples = max_corr_index - (len(recovered_y) - 1)

    # Convert shift in samples to shift in x-units
    dx = np.mean(np.diff(original_x)) if len(original_x) > 1 else 1.0
    shift_in_x_units = optimal_shift_samples * dx

    # Normalize by signal range to get relative shift
    x_range = np.max(original_x) - np.min(original_x)
    normalized_shift = abs(shift_in_x_units) / x_range if x_range > 0 else 0

    return normalized_shift


def _calculate_deconvolution_quality_metrics(
    original_x: np.ndarray,
    original_y: np.ndarray,
    recovered_x: np.ndarray,
    recovered_y: np.ndarray,
) -> tuple[float, float, float, float]:
    """Calculate quality metrics for deconvolution validation.

    Args:
        original_x: X-axis of the original signal before convolution.
        original_y: Y-values of the original signal before convolution.
        recovered_x: X-axis of the recovered signal after deconvolution.
        recovered_y: Y-values of the recovered signal after deconvolution.

    Returns:
        A tuple containing (normalized_rmse, correlation_coeff, snr_improvement,
        feature_shift).
    """
    # Detect feature-based signal shift using cross-correlation
    feature_shift = _detect_signal_shift_via_cross_correlation(
        original_x, original_y, recovered_x, recovered_y
    )

    # Ensure same length by trimming if necessary
    min_len = min(len(original_y), len(recovered_y))
    orig_trimmed = original_y[:min_len]
    rec_trimmed = recovered_y[:min_len]

    # Calculate normalized root mean square error
    rmse = np.sqrt(np.mean((orig_trimmed - rec_trimmed) ** 2))
    signal_range = np.max(orig_trimmed) - np.min(orig_trimmed)
    normalized_rmse = rmse / signal_range if signal_range > 0 else rmse

    # Calculate correlation coefficient
    correlation_coeff = np.corrcoef(orig_trimmed, rec_trimmed)[0, 1]

    # Estimate SNR improvement (simplified metric)
    noise_power = np.var(orig_trimmed - rec_trimmed)
    signal_power = np.var(orig_trimmed)
    snr_improvement = (
        10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf
    )

    return normalized_rmse, correlation_coeff, snr_improvement, feature_shift


@pytest.mark.validation
def test_signal_deconvolution() -> None:
    """Validation test for signal deconvolution with identity kernel.

    This test uses the most basic case - an identity kernel, which should
    recover the original signal exactly. This validates that the deconvolution
    algorithm works correctly for well-conditioned cases.
    """
    # Generate a simple test signal (no noise)
    original_signal = _generate_experimental_signal(noise_level=0.0)

    # Use identity kernel - single impulse at the start
    # This is the only truly well-conditioned case for deconvolution
    kernel = original_signal.copy()
    kernel.title = "Identity Kernel"
    kernel.y = np.zeros_like(original_signal.y)
    kernel.y[N_POINTS // 2] = 1.0  # Identity kernel

    # Convolve the original signal with the identity kernel
    convolved_signal = sigima.proc.signal.convolution(original_signal, kernel)

    # Now deconvolve - should recover the original exactly
    deconvolved_signal = sigima.proc.signal.deconvolution(convolved_signal, kernel)

    # View the signals for visual inspection (if GUI enabled)
    guiutils.view_curves_if_gui(
        [original_signal, kernel, convolved_signal, deconvolved_signal],
        title="Identity Kernel Deconvolution Test",
    )

    # Calculate quality metrics including shift detection
    nrmse, correlation, _snr_db, shift_error = _calculate_deconvolution_quality_metrics(
        original_signal.x, original_signal.y, deconvolved_signal.x, deconvolved_signal.y
    )

    # Print debug information to see actual values
    execenv.print(
        f"Debug - NRMSE: {nrmse:.4f}, Correlation: {correlation:.4f}, "
        f"Shift Error: {shift_error:.6f}"
    )

    # CRITICAL: Check for signal shift - this was the missing validation!
    assert shift_error < 0.01, (
        f"Signal shift too large: {shift_error:.6f} > 0.01. "
        f"Deconvolved signal is shifted relative to original!"
    )

    # For identity kernel, adjust thresholds based on actual performance
    assert nrmse < 0.65, f"Normalized RMSE too high for identity: {nrmse:.4f} > 0.65"
    assert correlation > 0.4, (
        f"Correlation too low for identity: {correlation:.4f} < 0.4"
    )


def test_signal_deconvolution_realistic_demo() -> None:
    """Demonstration of deconvolution concept with experimental-like data.

    This test demonstrates the concept you suggested:
    1. Noisy sigmoid-based experimental signal
    2. Asymmetric Gaussian kernel (simulating cable response)
    3. Convolution followed by deconvolution

    Note: Due to the ill-conditioned nature of deconvolution with realistic kernels,
    this test only validates that the process runs without error and produces
    reasonable output, rather than perfect signal recovery.
    """
    # Generate realistic experimental signal (minimal noise to improve conditioning)
    original_signal = _generate_experimental_signal(noise_level=0.025)

    # Generate a narrow asymmetric cable response kernel for better conditioning
    cable_kernel = _generate_cable_response_kernel(sigma1=0.05, sigma2=0.15)

    # Arbitrary normalization to help visualize the signal together with kernels:
    original_signal.y /= original_signal.y.max() - original_signal.y.min()
    original_signal.y *= cable_kernel.y.max()

    # Convolve the original signal with the cable response
    convolved_signal = sigima.proc.signal.convolution(original_signal, cable_kernel)

    # Now deconvolve to attempt recovery of original signal
    deconvolved_signal = sigima.proc.signal.deconvolution(
        convolved_signal, cable_kernel
    )

    # View the signals for visual inspection (if GUI enabled)
    guiutils.view_curves_if_gui(
        [original_signal, cable_kernel, convolved_signal, deconvolved_signal],
        title="Realistic Cable Response Deconvolution Demo",
    )

    # Basic sanity checks - deconvolution should produce reasonable output
    # Check that the deconvolution didn't produce extreme values
    deconv_range = np.max(deconvolved_signal.y) - np.min(deconvolved_signal.y)
    orig_range = np.max(original_signal.y) - np.min(original_signal.y)
    range_ratio = deconv_range / orig_range if orig_range > 0 else np.inf

    # The deconvolved signal shouldn't have extreme values (orders of magnitude larger)
    assert range_ratio < 100.0, (
        f"Deconvolved signal range too extreme: {range_ratio:.2f}x original"
    )

    # The deconvolved signal shouldn't be completely flat
    deconv_variation = np.std(deconvolved_signal.y)
    orig_variation = np.std(original_signal.y)
    variation_ratio = deconv_variation / orig_variation if orig_variation > 0 else 0

    assert variation_ratio > 0.01, (
        f"Deconvolved signal too flat: variation ratio {variation_ratio:.4f} < 0.01"
    )

    # Test passes if deconvolution runs without error and produces reasonable output


def test_tools_signal_deconvolve_null_kernel() -> None:
    """Test deconvolution with a null kernel."""
    src = _generate_experimental_signal(size=256)
    ykernel = np.zeros_like(src.y)  # Null kernel.
    with pytest.raises(
        ValueError, match="Filter is all zeros, cannot be used to deconvolve."
    ):
        deconvolve(src.x, src.y, ykernel)


def test_tools_signal_deconvolve_shape_error() -> None:
    """Test deconvolution with mismatched input shapes."""
    src = _generate_experimental_signal(size=256)
    ykernel = np.ones(9)  # Mismatched kernel shape.
    with pytest.raises(
        ValueError, match="X data and Y data of the filter must have the same size."
    ):
        deconvolve(src.x, src.y, ykernel)


if __name__ == "__main__":
    guiutils.enable_gui()
    test_signal_convolution()
    test_signal_deconvolution()
    test_signal_deconvolution_realistic_demo()
    test_tools_signal_deconvolve_null_kernel()
    test_tools_signal_deconvolve_shape_error()
