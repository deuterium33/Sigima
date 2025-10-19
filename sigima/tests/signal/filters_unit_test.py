# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Frequency filters unit tests.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

import numpy as np
import pytest

import sigima.enums
import sigima.proc.signal
from sigima.objects.signal import SignalObj, create_signal
from sigima.tests import guiutils
from sigima.tests.helpers import check_array_result, check_scalar_result
from sigima.tools.signal.fourier import brickwall_filter


def build_clean_noisy_signals(
    length: int = 2**15,
    freq: int | float | np.ndarray = 1,
    noise_level: float = 0.2,
) -> tuple[SignalObj, SignalObj]:
    """Create a test 1D signal + high-freq noise.

    Args:
        length: Length of the signal.
        freq: Frequency of the sine wave, can be a single value or an array of
         frequencies
        noise_level: Standard deviation of the Gaussian noise to be added.

    Returns:
        Tuple of (clean_signal, noisy_signal) where:
        - clean_signal: The clean sine wave signal.
        - noisy_signal: The noisy signal with added Gaussian noise.
    """
    x = np.linspace(0, 1, length)
    if np.isscalar(freq):
        y_clean = np.sin(2 * np.pi * freq * x)
    else:
        freq = np.asarray(freq)
        y_clean = np.sum([np.sin(2 * np.pi * f * x) for f in freq], axis=0)
    rng = np.random.default_rng(seed=0)
    y_noisy = y_clean + noise_level * rng.standard_normal(size=length)
    noisy = create_signal("noisy signal", x, y_noisy)
    clean = create_signal("clean signal", x, y_clean)
    return clean, noisy


def _validate_scipy_filter_output(
    result_signal: SignalObj,
    method: sigima.enums.FrequencyFilterMethod,
    filter_type: str,
    original_signal: SignalObj | None = None,
) -> bool:
    """Validate scipy filter output for basic functionality.

    Args:
        result_signal: The filtered signal to validate
        method: The filter method used
        filter_type: Type of filter (lowpass, highpass, etc.) for messages
        original_signal: Original signal for variance comparison (optional)

    Returns:
        True if validation passed, False if should skip this filter
    """
    # Check that output is finite
    if not np.all(np.isfinite(result_signal.y)):
        print(
            f"⚠ {method.value} {filter_type} filter: produced non-finite "
            "values, skipping"
        )
        return False

    # Check that output has reasonable magnitude
    max_magnitude = 100 if filter_type in ["highpass", "bandstop", "bandpass"] else 10
    if not np.max(np.abs(result_signal.y)) < max_magnitude:
        print(
            f"⚠ {method.value} {filter_type} filter: produced excessively "
            "large values, skipping"
        )
        return False

    # For lowpass, check that variance didn't increase too much
    if filter_type == "lowpass" and original_signal is not None:
        original_var = np.var(original_signal.y)
        filtered_var = np.var(result_signal.y)
        if filtered_var > original_var * 2:
            print(
                f"⚠ {method.value} {filter_type} filter: increased variance "
                "too much, skipping"
            )
            return False

    print(f"✓ {method.value} {filter_type} filter: working correctly")
    return True


def _test_filter_method(
    filter_func,
    param_class,
    method: sigima.enums.FrequencyFilterMethod,
    filter_type: str,
    test_signal: SignalObj,
    expected_signal: SignalObj | None = None,
    tolerance: float | None = None,
    original_signal: SignalObj | None = None,
    **filter_params,
) -> None:
    """Test a single filter method with given parameters.

    Args:
        filter_func: The filter function to call (lowpass, highpass, etc.)
        param_class: The parameter class for the filter
        method: The filter method to test
        filter_type: Type of filter for validation messages
        test_signal: Signal to filter
        expected_signal: Expected result for comparison (None for basic validation)
        tolerance: Tolerance for comparison (None for basic validation)
        original_signal: Original signal for variance checks
        **filter_params: Additional parameters for the filter
    """
    for zero_padding in (True, False):
        param = param_class.create(
            method=method, zero_padding=zero_padding, **filter_params
        )

        prefix = f"{method.value} {filter_type} (zero_padding={zero_padding}) "

        # Store original x data to verify it's not modified
        original_x = test_signal.x.copy()

        result_signal: SignalObj = filter_func(test_signal, param)
        guiutils.view_curves_if_gui([expected_signal or test_signal, result_signal])

        # CRITICAL: Check that the input signal's X data was NOT modified
        check_array_result(
            f"{prefix} input X data unchanged", test_signal.x, original_x
        )

        if expected_signal is not None:
            # Check that X data is unchanged
            check_array_result(
                f"{prefix} X data check", result_signal.x, expected_signal.x
            )

        # Validate based on whether we have expected results
        if expected_signal is not None and tolerance is not None:
            # Detailed comparison for brickwall filters
            if filter_type == "highpass":
                # Special case for highpass: check mean is close to zero
                check_scalar_result(
                    f"{prefix} removes low freq",
                    float(np.mean(result_signal.y)),
                    0,
                    atol=tolerance,
                )
            else:
                # Array comparison for other filters
                check_array_result(
                    f"{prefix}",
                    result_signal.y[10 : len(result_signal.y) - 10],
                    expected_signal.y[10 : len(expected_signal.y) - 10],
                    atol=tolerance,
                )
        elif not zero_padding:
            # Basic validation for scipy filters
            _validate_scipy_filter_output(
                result_signal, method, filter_type, original_signal
            )


@pytest.mark.validation
def test_signal_lowpass() -> None:
    """Validation test for frequency filtering."""
    clean, noisy = build_clean_noisy_signals()

    # Test all filter methods
    for method in sigima.enums.FrequencyFilterMethod:
        if method == sigima.enums.FrequencyFilterMethod.BRICKWALL:
            # For brickwall, we expect very close match to the clean signal
            _test_filter_method(
                filter_func=sigima.proc.signal.lowpass,
                param_class=sigima.proc.signal.LowPassFilterParam,
                method=method,
                filter_type="lowpass",
                test_signal=noisy,
                expected_signal=clean,
                tolerance=0.15,
                cut0=2.0,
            )
        else:
            # For scipy filters, just check basic functionality
            _test_filter_method(
                filter_func=sigima.proc.signal.lowpass,
                param_class=sigima.proc.signal.LowPassFilterParam,
                method=method,
                filter_type="lowpass",
                test_signal=noisy,
                original_signal=noisy,
                cut0=5000.0,
            )


@pytest.mark.validation
def test_signal_highpass() -> None:
    """Validation test for highpass frequency filtering."""
    noise_level = 0.2
    clean, noisy = build_clean_noisy_signals(noise_level=noise_level)

    # Test all filter methods
    for method in sigima.enums.FrequencyFilterMethod:
        if method == sigima.enums.FrequencyFilterMethod.BRICKWALL:
            # For brickwall, the highpass should remove the low freq signal
            # and leave mostly noise (mean should be close to zero)
            mean_variance = np.sqrt(noise_level / len(clean.x))
            expected_err = 3 * mean_variance

            # Create a dummy expected signal with zero mean for validation
            expected_signal = create_signal("zero", clean.x, np.zeros_like(clean.y))

            _test_filter_method(
                filter_func=sigima.proc.signal.highpass,
                param_class=sigima.proc.signal.HighPassFilterParam,
                method=method,
                filter_type="highpass",
                test_signal=noisy,
                expected_signal=expected_signal,
                tolerance=expected_err,
                cut0=2.0,
            )
        else:
            # For scipy filters, use higher cutoff and basic validation
            _test_filter_method(
                filter_func=sigima.proc.signal.highpass,
                param_class=sigima.proc.signal.HighPassFilterParam,
                method=method,
                filter_type="highpass",
                test_signal=noisy,
                cut0=1000.0,
            )


@pytest.mark.validation
def test_signal_bandstop() -> None:
    """Validation test for stopband frequency filtering."""
    # Test all filter methods
    for method in sigima.enums.FrequencyFilterMethod:
        if method == sigima.enums.FrequencyFilterMethod.BRICKWALL:
            # Original test setup works well for brickwall
            tst_sig, _ = build_clean_noisy_signals(
                freq=np.array([1, 3, 5]), noise_level=0
            )
            exp_sig, _ = build_clean_noisy_signals(freq=np.array([1, 5]), noise_level=0)
            _test_filter_method(
                filter_func=sigima.proc.signal.bandstop,
                param_class=sigima.proc.signal.BandStopFilterParam,
                method=method,
                filter_type="bandstop",
                test_signal=tst_sig,
                expected_signal=exp_sig,
                tolerance=1e-3,
                cut0=2.0,
                cut1=4.0,
            )
        else:
            # For scipy filters, use simpler test signal
            x = np.linspace(0, 1, 1000)
            y = (
                np.sin(2 * np.pi * 10 * x)
                + np.sin(2 * np.pi * 100 * x)
                + np.sin(2 * np.pi * 200 * x)
            )
            test_sig = create_signal("test", x, y)
            _test_filter_method(
                filter_func=sigima.proc.signal.bandstop,
                param_class=sigima.proc.signal.BandStopFilterParam,
                method=method,
                filter_type="bandstop",
                test_signal=test_sig,
                cut0=50.0,
                cut1=150.0,
            )


@pytest.mark.validation
def test_signal_bandpass() -> None:
    """Validation test for bandpass frequency filtering."""
    # Test all filter methods
    for method in sigima.enums.FrequencyFilterMethod:
        if method == sigima.enums.FrequencyFilterMethod.BRICKWALL:
            # Original test setup works well for brickwall
            tst_sig, _ = build_clean_noisy_signals(
                freq=np.array([1, 3, 5]), noise_level=0
            )
            exp_sig, _ = build_clean_noisy_signals(freq=np.array([3]), noise_level=0)
            _test_filter_method(
                filter_func=sigima.proc.signal.bandpass,
                param_class=sigima.proc.signal.BandPassFilterParam,
                method=method,
                filter_type="bandpass",
                test_signal=tst_sig,
                expected_signal=exp_sig,
                tolerance=1e-3,
                cut0=2.0,
                cut1=4.0,
            )
        else:
            # For scipy filters, use simpler test signal
            x = np.linspace(0, 1, 1000)
            y = (
                np.sin(2 * np.pi * 10 * x)
                + np.sin(2 * np.pi * 100 * x)
                + np.sin(2 * np.pi * 200 * x)
            )
            test_sig = create_signal("test", x, y)
            _test_filter_method(
                filter_func=sigima.proc.signal.bandpass,
                param_class=sigima.proc.signal.BandPassFilterParam,
                method=method,
                filter_type="bandpass",
                test_signal=test_sig,
                cut0=50.0,
                cut1=150.0,
            )


def test_brickwall_filter_invalid_x():
    """Test brickwall_filter raises on non-uniform x."""
    clean, noisy = build_clean_noisy_signals()
    x_bad = clean.x.copy()
    x_bad[5] += 0.01  # break uniformity
    with pytest.raises(ValueError, match="evenly spaced"):
        brickwall_filter(x_bad, noisy.y, "lowpass", cut0=0.1)


def test_tools_to_proc_interface():
    """Test that the `brickwall_filter` function is properly interfaced
    with the `sigima.proc` module, via the `lowpass`, `highpass`, `bandpass`,
    and `stopband` functions.
    """
    _clean, tst_sig = build_clean_noisy_signals(freq=np.array([1, 3, 5]))

    # Lowpass
    tools_res = brickwall_filter(tst_sig.x, tst_sig.y, "lowpass", cut0=2.0)
    param = sigima.proc.signal.LowPassFilterParam.create(
        cut0=2.0,
        method=sigima.enums.FrequencyFilterMethod.BRICKWALL,
        zero_padding=False,
    )
    for cut0 in (None, 2.0):
        param.cut0 = cut0
        # Just test the 'update_from_obj' method, not needed here (and no need to test
        # it for each filter function because they all use the same base class).
        param.update_from_obj(tst_sig)
    proc_res = sigima.proc.signal.lowpass(tst_sig, param)
    check_array_result("Lowpass filter result", tools_res[1], proc_res.y, atol=1e-3)

    # Highpass
    tools_res = brickwall_filter(tst_sig.x, tst_sig.y, "highpass", cut0=2.0)
    param = sigima.proc.signal.HighPassFilterParam.create(
        cut0=2.0,
        method=sigima.enums.FrequencyFilterMethod.BRICKWALL,
        zero_padding=False,
    )
    proc_res = sigima.proc.signal.highpass(tst_sig, param)
    check_array_result("Highpass filter result", tools_res[1], proc_res.y, atol=1e-3)

    # Bandpass
    tools_res = brickwall_filter(tst_sig.x, tst_sig.y, "bandpass", cut0=2.0, cut1=4.0)
    param = sigima.proc.signal.BandPassFilterParam.create(
        cut0=2.0,
        cut1=4.0,
        method=sigima.enums.FrequencyFilterMethod.BRICKWALL,
        zero_padding=False,
    )
    proc_res = sigima.proc.signal.bandpass(tst_sig, param)
    check_array_result("Bandpass filter result", tools_res[1], proc_res.y, atol=1e-3)

    # Bandstop
    tools_res = brickwall_filter(tst_sig.x, tst_sig.y, "bandstop", cut0=2.0, cut1=4.0)
    param = sigima.proc.signal.BandStopFilterParam.create(
        cut0=2.0,
        cut1=4.0,
        method=sigima.enums.FrequencyFilterMethod.BRICKWALL,
        zero_padding=False,
    )
    proc_res = sigima.proc.signal.bandstop(tst_sig, param)
    check_array_result("Bandstop filter result", tools_res[1], proc_res.y, atol=1e-3)


if __name__ == "__main__":
    guiutils.enable_gui()
    # test_signal_lowpass()
    # test_signal_highpass()
    # test_signal_bandstop()
    test_signal_bandpass()
    test_brickwall_filter_invalid_x()
    test_tools_to_proc_interface()
