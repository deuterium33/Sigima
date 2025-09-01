# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for uncertainty propagation in signal operations

Features from signal processing functions that include uncertainty propagation.
This test covers the mathematical functions (sqrt, log10, exp, clip, absolute,
real, imag) and arithmetic operations (addition, average, product, difference,
constant operations).
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np

import sigima.objects
import sigima.params
import sigima.proc.signal
import sigima.tests.data
from sigima.tests.helpers import check_array_result


def __create_signal_with_uncertainty() -> sigima.objects.SignalObj:
    """Create a signal with uncertainty data for testing."""
    obj = sigima.tests.data.create_periodic_signal(sigima.objects.SignalTypes.COSINUS)
    obj.dy = 0.1 * np.abs(obj.y) + 0.01  # 10% relative + 0.01 absolute
    return obj


def __create_signal_without_uncertainty() -> sigima.objects.SignalObj:
    """Create a signal without uncertainty data for testing."""
    obj = sigima.tests.data.create_periodic_signal(sigima.objects.SignalTypes.COSINUS)
    obj.dy = None
    return obj


def __verify_uncertainty_propagation(
    func: Callable[[sigima.objects.SignalObj], sigima.objects.SignalObj],
    param: sigima.params.GaussianParam
    | sigima.params.MovingAverageParam
    | sigima.params.MovingMedianParam
    | None = None,
) -> None:
    """Test uncertainty propagation for a given signal processing function."""
    src = __create_signal_with_uncertainty()
    if param is None:
        result = func(src)
    else:
        result = func(src, param)

    # Check that uncertainties are propagated (should be unchanged for filters)
    assert result.dy is not None, "Uncertainty should be propagated"
    check_array_result("Uncertainty propagation", result.dy, src.dy)

    # Test without uncertainty
    src = __create_signal_without_uncertainty()
    result_no_unc = func(src)
    assert result_no_unc.dy is None, (
        "Uncertainty should be None when input has no uncertainty"
    )


def test_sqrt_uncertainty_propagation() -> None:
    """Test uncertainty propagation for square root function."""
    # Test with uncertainty
    src = __create_signal_with_uncertainty()
    result = sigima.proc.signal.sqrt(src)

    # Check result values
    check_array_result("Square root values", result.y, np.sqrt(src.y))

    # Check uncertainty propagation: σ(√y) = σ(y) / (2√y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        expected_dy = src.dy / (2 * np.sqrt(src.y))
        expected_dy[np.isinf(expected_dy) | np.isnan(expected_dy)] = np.nan

    check_array_result("Square root uncertainty propagation", result.dy, expected_dy)

    # Test without uncertainty
    src_no_unc = __create_signal_without_uncertainty()
    result_no_unc = sigima.proc.signal.sqrt(src_no_unc)
    assert result_no_unc.dy is None, (
        "Uncertainty should be None when input has no uncertainty"
    )


def test_log10_uncertainty_propagation() -> None:
    """Test uncertainty propagation for log10 function."""
    # Test with uncertainty - use positive values to avoid log domain issues
    src = __create_signal_with_uncertainty()
    src.y = np.abs(src.y) + 1.0  # Ensure positive values
    result = sigima.proc.signal.log10(src)

    # Check result values
    check_array_result("Log10 values", result.y, np.log10(src.y))

    # Check uncertainty propagation: σ(log₁₀(y)) = σ(y) / (y * ln(10))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        expected_dy = src.dy / (src.y * np.log(10))
        expected_dy[np.isinf(expected_dy) | np.isnan(expected_dy)] = np.nan

    check_array_result("Log10 uncertainty propagation", result.dy, expected_dy)

    # Test without uncertainty
    src_no_unc = __create_signal_without_uncertainty()
    src_no_unc.y = np.abs(src_no_unc.y) + 1.0  # Ensure positive values
    result_no_unc = sigima.proc.signal.log10(src_no_unc)
    assert result_no_unc.dy is None, (
        "Uncertainty should be None when input has no uncertainty"
    )


def test_exp_uncertainty_propagation() -> None:
    """Test uncertainty propagation for exponential function."""
    # Test with uncertainty
    src = __create_signal_with_uncertainty()
    result = sigima.proc.signal.exp(src)

    # Check result values
    check_array_result("Exponential values", result.y, np.exp(src.y))

    # Check uncertainty propagation: σ(eʸ) = eʸ * σ(y) = dst.y * σ(y)
    expected_dy = np.abs(result.y) * src.dy
    check_array_result("Exponential uncertainty propagation", result.dy, expected_dy)

    # Test without uncertainty
    src_no_unc = __create_signal_without_uncertainty()
    result_no_unc = sigima.proc.signal.exp(src_no_unc)
    assert result_no_unc.dy is None, (
        "Uncertainty should be None when input has no uncertainty"
    )


def test_clip_uncertainty_propagation() -> None:
    """Test uncertainty propagation for clipping function."""
    # Test with uncertainty
    src = __create_signal_with_uncertainty()

    # Test clipping with both limits
    param = sigima.params.ClipParam.create(lower=-0.5, upper=0.5)
    result = sigima.proc.signal.clip(src, param)

    # Check result values
    expected_y = np.clip(src.y, param.lower, param.upper)
    check_array_result("Clip values", result.y, expected_y)

    # Check uncertainty propagation: σ(clip(y)) = σ(y) where not clipped,
    # 0 where clipped
    expected_dy = src.dy.copy()
    expected_dy[src.y <= param.lower] = 0
    expected_dy[src.y >= param.upper] = 0
    check_array_result("Clip uncertainty propagation", result.dy, expected_dy)

    # Test without uncertainty
    src_no_unc = __create_signal_without_uncertainty()
    result_no_unc = sigima.proc.signal.clip(src_no_unc, param)
    assert result_no_unc.dy is None, (
        "Uncertainty should be None when input has no uncertainty"
    )


def test_absolute_uncertainty_propagation() -> None:
    """Test uncertainty propagation for absolute value function."""
    __verify_uncertainty_propagation(sigima.proc.signal.absolute)


def test_real_uncertainty_propagation() -> None:
    """Test uncertainty propagation for real part function."""
    __verify_uncertainty_propagation(sigima.proc.signal.real)


def test_imag_uncertainty_propagation() -> None:
    """Test uncertainty propagation for imaginary part function."""
    # Test with uncertainty
    src = __create_signal_with_uncertainty()
    result = sigima.proc.signal.imag(src)

    # Check result values
    check_array_result("Imaginary part values", result.y, np.imag(src.y))

    # Check uncertainty propagation: uncertainties unchanged for imaginary part
    check_array_result("Imaginary part uncertainty propagation", result.dy, src.dy)

    # Test without uncertainty
    src_no_unc = __create_signal_without_uncertainty()
    result_no_unc = sigima.proc.signal.imag(src_no_unc)
    assert result_no_unc.dy is None, (
        "Uncertainty should be None when input has no uncertainty"
    )


def test_is_uncertainty_data_available() -> None:
    """Test the is_uncertainty_data_available helper function."""
    # Single signal with uncertainty
    src_with = __create_signal_with_uncertainty()
    assert sigima.proc.signal.is_uncertainty_data_available(src_with), (
        "Should return True for signal with uncertainty"
    )

    # Single signal without uncertainty
    src_without = __create_signal_without_uncertainty()
    assert not sigima.proc.signal.is_uncertainty_data_available(src_without), (
        "Should return False for signal without uncertainty"
    )

    # List of signals - all with uncertainty
    src_list_with = [__create_signal_with_uncertainty() for _ in range(3)]
    assert sigima.proc.signal.is_uncertainty_data_available(src_list_with), (
        "Should return True for list where all signals have uncertainty"
    )

    # List of signals - mixed
    src_list_mixed = [
        __create_signal_with_uncertainty(),
        __create_signal_without_uncertainty(),
    ]
    assert not sigima.proc.signal.is_uncertainty_data_available(src_list_mixed), (
        "Should return False for list with mixed uncertainty availability"
    )

    # List of signals - all without uncertainty
    src_list_without = [__create_signal_without_uncertainty() for _ in range(3)]
    assert not sigima.proc.signal.is_uncertainty_data_available(src_list_without), (
        "Should return False for list where no signals have uncertainty"
    )


def test_inverse_uncertainty_propagation() -> None:
    """Test uncertainty propagation for signal inversion."""
    # Test with uncertainty
    src = __create_signal_with_uncertainty()
    # Ensure values are not too close to zero to avoid division issues
    src.y = src.y + 2.0  # Shift away from zero
    result = sigima.proc.signal.inverse(src)

    # Check result values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        expected_y = 1.0 / src.y
        expected_y[np.isinf(expected_y)] = np.nan
    check_array_result("Inverse values", result.y, expected_y)

    # Check uncertainty propagation: σ(1/y) = |1/y| * σ(y) / |y| = σ(y) / |y|²
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        expected_dy = np.abs(result.y) * src.dy / np.abs(src.y)
        expected_dy[np.isinf(expected_dy)] = np.nan
    check_array_result("Inverse uncertainty propagation", result.dy, expected_dy)

    # Test without uncertainty
    src_no_unc = __create_signal_without_uncertainty()
    src_no_unc.y = src_no_unc.y + 2.0  # Shift away from zero
    result_no_unc = sigima.proc.signal.inverse(src_no_unc)
    assert result_no_unc.dy is None, (
        "Uncertainty should be None when input has no uncertainty"
    )


def test_gaussian_filter_uncertainty_propagation() -> None:
    """Test uncertainty propagation for Gaussian filter."""
    param = sigima.params.GaussianParam.create(sigma=2.0)
    __verify_uncertainty_propagation(sigima.proc.signal.gaussian_filter, param)


def test_wiener_filter_uncertainty_propagation() -> None:
    """Test uncertainty propagation for Wiener filter."""
    __verify_uncertainty_propagation(sigima.proc.signal.wiener)


def test_moving_average_uncertainty_propagation() -> None:
    """Test uncertainty propagation for moving average filter."""
    param = sigima.params.MovingAverageParam.create(n=5)
    __verify_uncertainty_propagation(sigima.proc.signal.moving_average, param)


def test_moving_median_uncertainty_propagation() -> None:
    """Test uncertainty propagation for moving median filter."""
    param = sigima.params.MovingMedianParam.create(n=5)
    __verify_uncertainty_propagation(sigima.proc.signal.moving_median, param)


def test_wrap1to1func_basic_behavior() -> None:
    """Test basic Wrap1to1Func behavior with uncertainty propagation.

    Wrap1to1Func should preserve uncertainty unchanged for any wrapped function.
    """
    # Test with a mathematical function (np.sqrt)
    # Note: This tests the wrapper behavior, not the direct sqrt function
    compute_sqrt_wrapped = sigima.proc.signal.Wrap1to1Func(np.sqrt)

    # Test with uncertainty
    src = __create_signal_with_uncertainty()
    result = compute_sqrt_wrapped(src)

    # Check result values
    check_array_result("Wrapped sqrt values", result.y, np.sqrt(src.y))

    # Check uncertainty propagation (should be unchanged when using Wrap1to1Func)
    check_array_result("Wrapped sqrt uncertainty propagation", result.dy, src.dy)

    # Test with a custom function
    def custom_multiply(y):
        """Custom function: multiply by 3."""
        return 3 * y

    compute_custom = sigima.proc.signal.Wrap1to1Func(custom_multiply)

    result_custom = compute_custom(src)

    # Check result values
    check_array_result("Custom multiply values", result_custom.y, 3 * src.y)

    # Check uncertainty propagation (should be unchanged for any wrapped function)
    check_array_result(
        "Custom multiply uncertainty propagation", result_custom.dy, src.dy
    )

    # Test without uncertainty
    src_no_unc = __create_signal_without_uncertainty()
    result_no_unc = compute_custom(src_no_unc)
    assert result_no_unc.dy is None, (
        "Uncertainty should be None when input has no uncertainty"
    )


def test_wrap1to1func_with_args_kwargs() -> None:
    """Test Wrap1to1Func with additional args and kwargs."""

    def power_func(y, power=2):
        """Raise y to a power."""
        return y**power

    # Test with power=3 using kwargs
    compute_power = sigima.proc.signal.Wrap1to1Func(power_func, power=3)

    src = __create_signal_with_uncertainty()
    result = compute_power(src)

    # Check result values
    check_array_result("Power 3 values", result.y, src.y**3)

    # Check uncertainty propagation (should be unchanged when using Wrap1to1Func)
    # Note: This is different from the mathematical uncertainty propagation
    # which would be σ(y³) = 3 * y² * σ(y)
    check_array_result("Power 3 uncertainty propagation", result.dy, src.dy)

    # Test with positional arguments
    def multiply_add(y, multiplier, addend):
        """Custom function: y * multiplier + addend."""
        return y * multiplier + addend

    compute_multiply_add = sigima.proc.signal.Wrap1to1Func(multiply_add, 2, addend=5)

    result_multiply_add = compute_multiply_add(src)

    # Check result values
    expected_y = src.y * 2 + 5
    check_array_result("Multiply-add values", result_multiply_add.y, expected_y)

    # Check uncertainty propagation (preserved unchanged)
    check_array_result(
        "Multiply-add uncertainty propagation", result_multiply_add.dy, src.dy
    )


if __name__ == "__main__":
    test_sqrt_uncertainty_propagation()
    test_log10_uncertainty_propagation()
    test_exp_uncertainty_propagation()
    test_clip_uncertainty_propagation()
    test_absolute_uncertainty_propagation()
    test_real_uncertainty_propagation()
    test_imag_uncertainty_propagation()
    test_is_uncertainty_data_available()
    test_inverse_uncertainty_propagation()
    test_gaussian_filter_uncertainty_propagation()
    test_wiener_filter_uncertainty_propagation()
    test_moving_average_uncertainty_propagation()
    test_moving_median_uncertainty_propagation()
    test_wrap1to1func_basic_behavior()
    test_wrap1to1func_with_args_kwargs()
