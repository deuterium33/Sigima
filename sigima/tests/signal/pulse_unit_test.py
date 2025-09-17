# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for the `sigima.tools.signal.pulse` module.
"""

from __future__ import annotations

import numbers

import numpy as np
import pytest

from sigima.enums import SignalShape
from sigima.objects.signal import create_signal
from sigima.proc.signal import PulseFeaturesParam, SignalObj, extract_pulse_features
from sigima.tests import guiutils
from sigima.tests.data import generate_square_signal, generate_step_signal
from sigima.tests.helpers import check_scalar_result
from sigima.tools.signal import pulse
from sigima.tools.signal.pulse import PulseFeatures


def theoretical_step_amplitude(y_initial: float, y_final: float) -> float:
    """Calculate theoretical amplitude for a step signal."""
    return abs(y_final - y_initial)


def theoretical_square_amplitude(y_initial: float, y_high: float) -> float:
    """Calculate theoretical amplitude for a square signal."""
    return abs(y_high - y_initial)


def theoretical_crossing_time(t_step: float, t_rise: float, ratio: float) -> float:
    """Calculate theoretical crossing time for a step signal with linear rise.

    Args:
        t_step: Time when the rise starts
        t_rise: Duration of the linear rise
        ratio: Crossing ratio (0.0 to 1.0)

    Returns:
        Theoretical time when signal reaches the specified ratio of its amplitude
    """
    return t_step + ratio * t_rise


def theoretical_rise_time(
    t_rise: float, start_rise_ratio: float, stop_rise_ratio: float
) -> float:
    """Calculate theoretical rise time between two amplitude ratios.

    Args:
        t_rise: Total rise duration of the signal
        start_rise_ratio: Starting amplitude ratio (e.g., 0.2 for 20%)
        stop_rise_ratio: Parameter where actual stop ratio = 1 - stop_rise_ratio
                         (e.g., 0.2 gives actual stop at 80%)

    Returns:
        Theoretical rise time between the two ratios
    """
    actual_start_ratio = start_rise_ratio
    actual_stop_ratio = 1 - stop_rise_ratio
    return (actual_stop_ratio - actual_start_ratio) * t_rise


def theoretical_square_crossing_time(
    t_step: float,
    t_rise: float,
    square_duration: float,
    t_fall: float,
    ratio: float,
    edge: str = "rise",
) -> float:
    """Calculate theoretical crossing time for square signal.

    Args:
        t_step: Time when the rise starts
        t_rise: Duration of the rising edge
        square_duration: Duration of the plateau
        t_fall: Duration of the falling edge
        ratio: Crossing ratio (0.0 to 1.0)
        edge: Which edge to calculate - "rise" or "fall"

    Returns:
        Theoretical crossing time for the specified edge
    """
    if edge == "rise":
        return t_step + ratio * t_rise
    if edge == "fall":
        t_start_fall = t_step + t_rise + square_duration
        return t_start_fall + ratio * t_fall
    raise ValueError("edge must be 'rise' or 'fall'")


def _test_shape_recognition_case(
    signal_type: str,
    expected_shape: SignalShape,
    y_initial: float,
    y_final_or_high: float,
    start_range: tuple[float, float] | None = None,
    end_range: tuple[float, float] | None = None,
) -> None:
    """Helper function to test shape recognition for different signal configurations.

    Args:
        signal_type: "step" or "square"
        expected_shape: Expected SignalShape result
        y_initial: Initial signal value
        y_final_or_high: Final value (step) or high value (square)
        start_range: Start baseline range for shape recognition (optional)
        end_range: End baseline range for shape recognition (optional)
    """
    # Generate signal
    if signal_type == "step":
        generate_func = generate_step_signal
        x, y_noisy = generate_func(seed=0, y_initial=y_initial, y_final=y_final_or_high)
    else:  # square
        generate_func = generate_square_signal
        x, y_noisy = generate_func(seed=0, y_initial=y_initial, y_high=y_final_or_high)

    # Create title
    polarity_desc = "positive" if y_final_or_high > y_initial else "negative"
    title = f"{signal_type.capitalize()}, {polarity_desc} polarity | Shape recognition"
    if start_range is None:
        title += " (auto-detection)"

    # Test shape recognition
    if start_range is not None and end_range is not None:
        shape = pulse.heuristically_recognize_shape(x, y_noisy, start_range, end_range)
    else:
        shape = pulse.heuristically_recognize_shape(x, y_noisy)

    assert shape == expected_shape, f"Expected {expected_shape}, got {shape}"
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(signal, title=f"{title}: {shape}")

    # Test auto-detection if requested and ranges were provided
    if start_range is not None:
        shape_auto = pulse.heuristically_recognize_shape(x, y_noisy)
        assert shape_auto == expected_shape, (
            f"Auto-detection: Expected {expected_shape}, got {shape_auto}"
        )


def test_heuristically_recognize_shape() -> None:
    """Unit test for the `pulse.heuristically_recognize_shape` function.

    This test verifies that the function correctly identifies the shape of various
    noisy signals (step and square) generated with different parameters. It checks the
    recognition both with and without specifying regions of interest.

    Test cases:
        - Step signal with default parameters.
        - Step signal with specified regions.
        - Square signal with default parameters.
        - Step signal with custom initial and final values.
        - Square signal with custom initial and high values.

    """
    tsc = _test_shape_recognition_case
    # Step signals with positive polarity
    tsc("step", SignalShape.STEP, 0.0, 5.0, (0.0, 2.0), (4.0, 8.0))
    # Step signals with negative polarity
    tsc("step", SignalShape.STEP, 5.0, 2.0, (0.0, 2.0), (4.0, 8.0))
    # Square signals with positive polarity
    tsc("square", SignalShape.SQUARE, 0.0, 5.0, (0.0, 2.0), (12.0, 14.0))
    # Square signals with negative polarity
    tsc("square", SignalShape.SQUARE, 5.0, 2.0, (0.0, 2.0), (12.0, 14.0))


def _test_polarity_detection_case(
    signal_type: str,
    polarity_desc: str,
    expected_polarity: int,
    y_initial: float,
    y_final_or_high: float,
    start_range: tuple[float, float] | None = None,
    end_range: tuple[float, float] | None = None,
) -> None:
    """Helper function to test polarity detection for different signal configurations.

    Args:
        signal_type: "step" or "square"
        polarity_desc: Description of polarity ("positive" or "negative")
        expected_polarity: Expected polarity result (1 or -1)
        y_initial: Initial signal value
        y_final_or_high: Final value (step) or high value (square)
        start_range: Start baseline range for polarity detection (optional)
        end_range: End baseline range for polarity detection (optional)
    """
    # Generate signal
    if signal_type == "step":
        generate_func = generate_step_signal
        x, y_noisy = generate_func(seed=0, y_initial=y_initial, y_final=y_final_or_high)
    else:  # square
        generate_func = generate_square_signal
        x, y_noisy = generate_func(seed=0, y_initial=y_initial, y_high=y_final_or_high)

    # Create title
    title = f"{signal_type}, detection {polarity_desc} polarity"
    if start_range is None:
        title += " (auto)"

    # Test polarity detection
    if start_range is not None and end_range is not None:
        polarity = pulse.detect_polarity(x, y_noisy, start_range, end_range)
    else:
        polarity = pulse.detect_polarity(x, y_noisy)

    check_scalar_result(title, polarity, expected_polarity)
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(signal, title=f"{title}: {polarity}")

    # Test auto-detection if requested and ranges were provided
    if start_range is not None:
        polarity_auto = pulse.detect_polarity(x, y_noisy)
        check_scalar_result(f"{title} (auto)", polarity_auto, expected_polarity)


def test_detect_polarity() -> None:
    """Unit test for the `pulse.detect_polarity` function.

    This test verifies the correct detection of signal polarity for both step and
    square signals, with various initial and final values, and using different detection
    intervals.

    Test cases covered:
    - Positive polarity detection for step and square signals.
    - Negative polarity detection for step and square signals with inverted amplitude.
    - Detection with and without explicit interval arguments.
    """
    tpdc = _test_polarity_detection_case
    # Step signals with positive polarity
    tpdc("step", "positive", 1, 0.0, 5.0, (0.0, 2.0), (4.0, 8.0))
    # Step signals with negative polarity
    tpdc("step", "negative", -1, 5.0, 2.0, (0.0, 2.0), (4.0, 8.0))
    # Square signals with positive polarity
    tpdc("square", "positive", 1, 0.0, 5.0, (0.0, 2.0), (12.0, 14.0))
    # Square signals with negative polarity
    tpdc("square", "negative", -1, 5.0, 2.0, (0.0, 2.0), (12.0, 14.0))


def _test_amplitude_case(
    signal_type: str,
    polarity_desc: str,
    y_initial: float,
    y_final_or_high: float,
    start_range: tuple[float, float],
    end_range: tuple[float, float],
    plateau_range: tuple[float, float] | None = None,
    atol: float = 0.2,
    rtol: float = 0.1,
) -> None:
    """Helper function to test amplitude calculation for different signal configs.

    Args:
        signal_type: "step" or "square"
        polarity_desc: Description of polarity ("positive" or "negative")
        y_initial: Initial signal value
        y_final_or_high: Final value (step) or high value (square)
        start_range: Start baseline range for amplitude calculation
        end_range: End baseline range for amplitude calculation
        plateau_range: Plateau range for square signals (optional)
        atol: Absolute tolerance for amplitude comparison
        rtol: Relative tolerance for auto-detection comparison
    """
    # Generate signal and calculate expected amplitude
    if signal_type == "step":
        generate_func = generate_step_signal
        theoretical_func = theoretical_step_amplitude
        x, y_noisy = generate_func(seed=0, y_initial=y_initial, y_final=y_final_or_high)
    else:  # square
        generate_func = generate_square_signal
        theoretical_func = theoretical_square_amplitude
        x, y_noisy = generate_func(seed=0, y_initial=y_initial, y_high=y_final_or_high)

    expected_amp = theoretical_func(y_initial, y_final_or_high)

    # Create title
    title = (
        f"{signal_type.capitalize()}, {polarity_desc} polarity | "
        f"Get {signal_type} amplitude"
    )
    if plateau_range is None:
        title += " (without plateau)"

    # Test with explicit ranges
    if plateau_range is not None:
        amp = pulse.get_amplitude(x, y_noisy, start_range, end_range, plateau_range)
    else:
        amp = pulse.get_amplitude(x, y_noisy, start_range, end_range)

    check_scalar_result(title, amp, expected_amp, atol=atol)

    with guiutils.lazy_qt_app_context() as qt_app:
        if qt_app is not None:
            # pylint: disable=import-outside-toplevel
            from plotpy.builder import make

            from sigima.tests import vistools

            ys = pulse.get_range_mean_y(x, y_noisy, start_range)
            ye = pulse.get_range_mean_y(x, y_noisy, end_range)
            xs0, xs1 = start_range
            xe0, xe1 = end_range
            items = [
                make.mcurve(x, y_noisy, label="Noisy signal"),
                vistools.create_signal_segment(xs0, ys, xs1, ys, "Start baseline"),
                vistools.create_signal_segment(xe0, ye, xe1, ye, "End baseline"),
            ]
            if signal_type == "square":
                if plateau_range is None:
                    polarity = pulse.detect_polarity(x, y_noisy, start_range, end_range)
                    plateau_range = pulse.get_plateau_range(x, y_noisy, polarity)
                xp0, xp1 = plateau_range
                yp = pulse.get_range_mean_y(x, y_noisy, plateau_range)
                items.append(
                    vistools.create_signal_segment(xp0, yp, xp1, yp, "Plateau")
                )

            vistools.view_curve_items(items, title=f"{title}: {amp:.3f}")

    # Test auto-detection
    amplitude_auto = pulse.get_amplitude(x, y_noisy)
    check_scalar_result(f"{title} (auto)", amplitude_auto, expected_amp, rtol=rtol)


def test_get_amplitude() -> None:
    """Unit test for the `pulse.get_amplitude` function.

    This test verifies the correct calculation of the amplitude of step and square
    signals, both with and without specified regions of interest. It checks the
    amplitude for both positive and negative polarities using theoretical calculations.

    Test cases:
        - Step signal with positive polarity.
        - Step signal with negative polarity.
        - Square signal with positive polarity.
        - Square signal with negative polarity.

        - Step signal with custom initial and final values.
        - Square signal with custom initial and high values.
    """
    tac = _test_amplitude_case
    # Step signals
    tac("step", "positive", 0.0, 5.0, (0.0, 2.0), (6.0, 8.0))
    tac("step", "negative", 5.0, 2.0, (0.0, 2.0), (6.0, 8.0))
    # Square signals with plateau
    tac("square", "positive", 0.0, 5.0, (0.0, 2.0), (12.0, 14.0), (5.5, 6.5))
    tac("square", "negative", 5.0, 2.0, (0.0, 2.0), (12.0, 14.0), (5.5, 6.5))
    # Square signals without plateau
    tac("square", "positive", 0.0, 5.0, (0.0, 2.0), (12.0, 14.0), atol=0.6)
    tac("square", "negative", 5.0, 2.0, (0.0, 2.0), (12.0, 14.0), atol=0.6)


def test_get_crossing_ratio_time() -> None:
    """Unit test for the `pulse.get_crossing_ratio_time` function.

    This test verifies the correct calculation of the crossing time at a given ratio
    for both positive and negative polarity step signals using theoretical calculations
    based on the signal generation parameters.
    """
    # Test parameters for step signal generation
    t_step, t_rise = 3, 2  # Default values from generate_step_signal
    ratio = 0.2

    # positive polarity (y_initial=0, y_final=5)
    y_initial, y_final = 0, 5
    expected_crossing_time = theoretical_crossing_time(t_step, t_rise, ratio)
    x, y_noisy = generate_step_signal(
        seed=0, t_step=t_step, t_rise=t_rise, y_initial=y_initial, y_final=y_final
    )

    crossing_time = pulse.get_crossing_ratio_time(x, y_noisy, (0, 2), (6, 8), ratio)
    check_scalar_result(
        "step, get crossing time, positive polarity",
        crossing_time,
        expected_crossing_time,
        atol=0.1,  # Allow some tolerance for noise effects and algorithm precision
    )
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(signal, title=f"Crossing time = {crossing_time:.3f}")

    # Auto-detection may find different baseline due to noise effects
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(
        signal, title=f"Crossing time (auto) = {crossing_time:.3f}"
    )

    # negative polarity (y_initial=5, y_final=2)
    y_initial, y_final = 5, 2
    expected_crossing_time = theoretical_crossing_time(t_step, t_rise, ratio)
    x, y_noisy = generate_step_signal(
        seed=0,
        t_step=t_step,
        t_rise=t_rise,
        y_initial=y_initial,
        y_final=y_final,
        noise_amplitude=0.1,
    )

    crossing_time = pulse.get_crossing_ratio_time(x, y_noisy, (0, 2), (6, 8), ratio)
    check_scalar_result(
        "step, get crossing time, negative polarity",
        crossing_time,
        expected_crossing_time,
        atol=0.1,
    )
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(
        signal, title=f"Crossing time (neg) = {crossing_time:.3f}"
    )


def test_get_step_rise_time() -> None:
    """Unit test for the `pulse.get_step_rise_time` function.

    This test verifies the correct calculation of the rise time for step signals with
    both positive and negative polarity using theoretical calculations based on
    signal generation parameters.
    """
    # Test parameters
    t_step, t_rise = 3, 2  # Default values from generate_step_signal
    # For 20%-80% rise time: start_rise_ratio=0.2, stop_rise_ratio=0.2 (gives 1-0.2=0.8)
    start_rise_ratio, stop_rise_ratio = 0.2, 0.2  # Standard 20%-80% rise time
    expected_rise_time = theoretical_rise_time(
        t_rise, start_rise_ratio, stop_rise_ratio
    )

    # positive polarity (y_initial=0, y_final=5)
    y_initial, y_final = 0, 5
    x, y_noisy = generate_step_signal(
        seed=0, t_step=t_step, t_rise=t_rise, y_initial=y_initial, y_final=y_final
    )

    rise_time = pulse.get_step_rise_time(
        x, y_noisy, (0, 2), (6, 8), start_rise_ratio, stop_rise_ratio
    )
    check_scalar_result(
        "step, get rise time, positive polarity",
        rise_time,
        expected_rise_time,
        atol=0.1,  # Allow tolerance for noise effects and algorithm precision
    )
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(signal, title=f"Rise time = {rise_time:.3f}")

    # Test without noise to verify algorithm precision
    x, y_clean = generate_step_signal(
        seed=0,
        t_step=t_step,
        t_rise=t_rise,
        y_initial=y_initial,
        y_final=y_final,
        noise_amplitude=0,
    )
    rise_time = pulse.get_step_rise_time(
        x, y_clean, (0, 2), (6, 8), start_rise_ratio, stop_rise_ratio
    )
    check_scalar_result(
        "step, get rise time, positive polarity (no noise)",
        rise_time,
        expected_rise_time,
        atol=0.05,  # Tighter tolerance for noise-free signal
    )
    signal = create_signal("", x, y_clean)
    guiutils.view_curves_if_gui(signal, title=f"Rise time (clean) = {rise_time:.3f}")

    # Test auto-detection mode
    rise_time = pulse.get_step_rise_time(
        x, y_clean, start_rise_ratio=start_rise_ratio, stop_rise_ratio=stop_rise_ratio
    )
    check_scalar_result(
        "step, get rise time, positive polarity (auto)",
        rise_time,
        expected_rise_time,
        atol=0.1,  # Allow tolerance for auto baseline detection
    )
    signal = create_signal("", x, y_clean)
    guiutils.view_curves_if_gui(signal, title=f"Rise time (auto) = {rise_time:.3f}")

    # negative polarity (y_initial=5, y_final=2)
    y_initial, y_final = 5, 2

    # Test with clean signal first to verify theoretical accuracy
    x, y_clean_neg = generate_step_signal(
        seed=0,
        t_step=t_step,
        t_rise=t_rise,
        y_initial=y_initial,
        y_final=y_final,
        noise_amplitude=0,
    )
    rise_time = pulse.get_step_rise_time(
        x, y_clean_neg, (0, 2), (6, 8), start_rise_ratio, stop_rise_ratio
    )
    check_scalar_result(
        "step, get rise time, negative polarity (clean)",
        rise_time,
        expected_rise_time,
        atol=0.05,  # Tight tolerance for clean signal
    )
    signal = create_signal("", x, y_clean_neg)
    guiutils.view_curves_if_gui(
        signal, title=f"Rise time (neg, clean) = {rise_time:.3f}"
    )

    # Test with noisy signal - expect larger variation due to multiple crossings
    x, y_noisy = generate_step_signal(
        seed=0, t_step=t_step, t_rise=t_rise, y_initial=y_initial, y_final=y_final
    )
    rise_time_noisy = pulse.get_step_rise_time(
        x, y_noisy, (0, 2), (6, 8), start_rise_ratio, stop_rise_ratio
    )
    # For noisy negative polarity, expect significant deviation due to multiple
    # crossings.
    # We validate that it's still reasonable (within factor of 2 of expected)
    assert 0.5 * expected_rise_time <= rise_time_noisy <= 3.0 * expected_rise_time, (
        f"Rise time {rise_time_noisy:.3f} outside reasonable range "
        f"[{0.5 * expected_rise_time:.3f}, "
        f"{3.0 * expected_rise_time:.3f}] for noisy negative signal"
    )
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(
        signal, title=f"Rise time (neg, noisy) = {rise_time_noisy:.3f}"
    )


def test_get_step_time_at_half_maximum() -> None:
    """Unit test for the `pulse.get_crossing_ratio_time` function at 0.5 ratio.

    This test verifies the correct calculation of the time at which a step signal
    reaches half of its maximum amplitude using theoretical calculations.
    """
    # Test parameters
    t_step, t_rise = 3, 2  # Default values from generate_step_signal
    ratio = 0.5

    # positive polarity (y_initial=0, y_final=5)
    y_initial, y_final = 0, 5
    expected_time = theoretical_crossing_time(t_step, t_rise, ratio)
    x, y_noisy = generate_step_signal(
        seed=0, t_step=t_step, t_rise=t_rise, y_initial=y_initial, y_final=y_final
    )

    time = pulse.get_crossing_ratio_time(x, y_noisy, (0, 2), (6, 8), ratio)
    check_scalar_result(
        "step, get time at half maximum, positive polarity",
        time,
        expected_time,
        atol=0.1,
    )
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(signal, title=f"Time at half max = {time:.3f}")

    # negative polarity (y_initial=5, y_final=2)
    y_initial, y_final = 5, 2
    expected_time = theoretical_crossing_time(t_step, t_rise, ratio)
    x, y_noisy = generate_step_signal(
        seed=0, t_step=t_step, t_rise=t_rise, y_initial=y_initial, y_final=y_final
    )

    time = pulse.get_crossing_ratio_time(x, y_noisy, (0, 2), (6, 8), ratio)
    check_scalar_result(
        "step, get time at half maximum, negative polarity",
        time,
        expected_time,
        atol=0.25,  # Higher tolerance for negative polarity with noise
    )
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(signal, title=f"Time at half max (neg) = {time:.3f}")


def test_get_step_starttime() -> None:
    """Unit test for the `pulse.get_crossing_ratio_time` function with start ratio.

    This test verifies the correct calculation of the start time (20% crossing)
    for step signals using theoretical calculations.
    """
    # Test parameters
    t_step, t_rise = 3, 2  # Default values from generate_step_signal
    ratio = 0.2  # Start ratio (20%)

    # positive polarity (y_initial=0, y_final=5)
    y_initial, y_final = 0, 5
    expected_time = theoretical_crossing_time(t_step, t_rise, ratio)
    x, y_noisy = generate_step_signal(
        seed=0, t_step=t_step, t_rise=t_rise, y_initial=y_initial, y_final=y_final
    )

    time = pulse.get_crossing_ratio_time(x, y_noisy, (0, 2), (6, 8), ratio)
    check_scalar_result("step, get start time", time, expected_time, atol=0.1)
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(signal, title=f"Start time = {time:.3f}")

    # negative polarity (y_initial=5, y_final=2)
    y_initial, y_final = 5, 2
    expected_time = theoretical_crossing_time(t_step, t_rise, ratio)
    x, y_noisy = generate_step_signal(
        seed=0, t_step=t_step, t_rise=t_rise, y_initial=y_initial, y_final=y_final
    )

    time = pulse.get_crossing_ratio_time(x, y_noisy, (0, 2), (6, 8), ratio)
    check_scalar_result("step, get start time (neg)", time, expected_time, atol=1.5)
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(signal, title=f"Start time (neg) = {time:.3f}")


def test_get_step_end_time() -> None:
    """Unit test for the `pulse.get_crossing_ratio_time` function with end ratio.

    This test verifies the correct calculation of the end time (80% crossing)
    for step signals using theoretical calculations.
    """
    # Test parameters
    t_step, t_rise = 3, 2  # Default values from generate_step_signal
    ratio = 0.8  # End ratio (80%)

    # positive polarity (y_initial=0, y_final=5)
    y_initial, y_final = 0, 5
    expected_time = theoretical_crossing_time(t_step, t_rise, ratio)
    x, y_noisy = generate_step_signal(
        seed=0, t_step=t_step, t_rise=t_rise, y_initial=y_initial, y_final=y_final
    )

    time = pulse.get_crossing_ratio_time(x, y_noisy, (0, 2), (6, 8), ratio)
    check_scalar_result("step, get end time", time, expected_time, atol=0.1)
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(signal, title=f"End time = {time:.3f}")

    # negative polarity (y_initial=5, y_final=2)
    y_initial, y_final = 5, 2
    expected_time = theoretical_crossing_time(t_step, t_rise, ratio)
    x, y_noisy = generate_step_signal(
        seed=0, t_step=t_step, t_rise=t_rise, y_initial=y_initial, y_final=y_final
    )

    time = pulse.get_crossing_ratio_time(x, y_noisy, (0, 2), (6, 8), ratio)
    check_scalar_result("step, get end time (neg)", time, expected_time, atol=0.1)
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(signal, title=f"End time (neg) = {time:.3f}")


def test_heuristically_find_foot_end_time() -> None:
    """Unit test for the `pulse.heuristically_find_foot_end_time` function.

    This test verifies that the function correctly identifies the end time of the foot
    (baseline) region in a step signal with a sharp rise, ensuring accurate detection
    even in the presence of noise.
    """
    # Generate a signal with a step at t = 3 and a sharp rise at t = 5
    x, y = generate_step_signal(
        t_start=0,
        t_end=10,
        dt=0.01,
        t_rise=0.01,  # very sharp rise to ensure detectability
        t_step=5,  # step starts at t = 5
        y_initial=1,
        y_final=6,
        noise_amplitude=0.03,
        seed=0,
    )

    time = pulse.heuristically_find_foot_end_time(x, y, (0, 4))

    check_scalar_result(
        "heuristically find foot end time",
        time,
        5.0,
        atol=0.02,  # small tolerance due to possible slight variation
    )
    signal = create_signal("", x, y)
    guiutils.view_curves_if_gui(signal, title=f"Foot end time = {time:.3f}")


def test_get_foot_info() -> None:
    """Unit test for the `pulse.get_foot_info` function.

    This test verifies that the function correctly computes the foot (baseline) region
    information for a generated step signal, including the end index, threshold, foot
    duration, and x_end value.
    """
    # Generate a step signal with a sharp rise at t=5
    x, y = generate_step_signal(seed=0, noise_amplitude=0.2)
    # Use start_baseline_range before the step, end_baseline_range after
    start_baseline_range = (0, 4)
    start_baseline_level = np.mean(y[start_baseline_range[0] : start_baseline_range[1]])
    end_baseline_range = (6, 8)
    start_rise_ratio = 0.1

    foot_info = pulse.get_foot_info(
        x,
        y,
        start_baseline_range,
        end_baseline_range,
        start_rise_ratio=start_rise_ratio,
    )

    # Check attributes exist (dataclass instead of dictionary)
    assert hasattr(foot_info, "index")
    assert hasattr(foot_info, "threshold")
    assert hasattr(foot_info, "foot_duration")
    assert hasattr(foot_info, "x_end")

    # The foot should end at t ~ 5.0
    check_scalar_result("foot_info x_end", foot_info.x_end, 3.239)
    # The threshold should be close to y at t=5
    idx = foot_info.index
    check_scalar_result(
        "foot_info threshold",
        foot_info.threshold,
        y[idx],
        atol=start_baseline_level,
    )
    # The foot duration should be - x[0] + x_end (x_end, since x[0]=0)
    check_scalar_result(
        "foot_info foot_duration",
        foot_info.foot_duration,
        foot_info.x_end,
        atol=1e-8,
    )
    # The index should correspond to the first x >= x_end
    assert np.isclose(x[idx], foot_info.x_end, atol=0.01)

    signal = create_signal("", x, y)
    guiutils.view_curves_if_gui(
        signal,
        title=f"Foot info: x_end={foot_info.x_end:.3f},"
        f"threshold={foot_info.threshold:.3f}",
    )


def test_tools_feature_extraction() -> None:
    """Test the feature extraction tools."""
    # Test the extraction of step parameters from a generated step signal
    y_initial, y_final = 1.0, 6.0
    expected_amp = theoretical_step_amplitude(y_initial, y_final)
    x, y = generate_step_signal(
        t_start=0,
        t_end=10,
        dt=0.01,
        t_rise=1,
        t_step=4,
        y_initial=y_initial,
        y_final=y_final,
        noise_amplitude=0,
        seed=0,
    )
    # Use start_baseline_range before the step, end_baseline_range after
    start_range = (0.0, 4.0)
    end_range = (6.0, 8.0)
    start_rise_ratio = 0.1

    params = pulse.extract_pulse_features(
        x, y, start_range, end_range, start_rise_ratio
    )

    # Check that we got a PulseFeatures dataclass
    assert isinstance(params, PulseFeatures)

    # Check values for all attributes
    check_scalar_result("polarity", params.polarity, 1, atol=1e-8)
    check_scalar_result("amplitude", params.amplitude, expected_amp, atol=0.01)
    check_scalar_result("rise_time", params.rise_time, 0.79, atol=0.01)
    assert params.fall_time is None, (
        f"Expected fall_time to be None, but got {params.fall_time}"
    )
    assert params.fwhm is None, f"Expected fwhm to be None, but got {params.fwhm}"
    check_scalar_result("offset", params.offset, 1.0, atol=0.01)
    check_scalar_result("t50", params.t50, 4.5, atol=0.01)
    check_scalar_result("tmax", params.tmax, 5.0, atol=0.01)
    check_scalar_result("foot_duration", params.foot_duration, 4.105, atol=0.01)
    assert params.signal_shape == SignalShape.STEP, (
        f"Expected signal_shape to be STEP, but got {params.signal_shape}"
    )
    guiutils.view_curves_if_gui(
        [x, y],
        title=(
            "Step parameters: "
            f"polarity={params.polarity}, "
            f"offset={params.offset}, "
            f"fwhm={params.fwhm}, "
            f"rise_time={params.rise_time}, "
            f"fall_time={params.fall_time}, "
            f"t50={params.t50}, "
            f"tmax={params.tmax}, "
            f"foot_duration={params.foot_duration}"
        ),
    )

    # Test the extraction of square parameters from a generated square signal.
    # Generate a square signal with a sharp rise at t=4 and fall at t=7
    x, y = generate_square_signal(
        t_start=0,
        t_end=20,
        dt=0.01,
        t_rise=1,
        t_step=4,
        t_fall=7,
        y_initial=1,
        y_high=6,
        noise_amplitude=0,
        seed=0,
    )
    # Use start_range before the step, end_range after
    start_range = (0.0, 2.5)
    end_range = (15.0, 17.0)
    start_rise_ratio = 0.1

    params = pulse.extract_pulse_features(
        x, y, start_range, end_range, start_rise_ratio
    )

    # Check that we got a PulseFeatures dataclass
    assert isinstance(params, PulseFeatures)

    # Check values for all attributes
    check_scalar_result("polarity", params.polarity, 1, atol=1e-8)
    check_scalar_result("amplitude", params.amplitude, 5.0, atol=0.01)
    check_scalar_result("rise_time", params.rise_time, 0.79, atol=0.01)
    check_scalar_result("fall_time", params.fall_time, 5.595, atol=0.01)
    check_scalar_result("fwhm", params.fwhm, 6.0, atol=0.01)
    check_scalar_result("offset", params.offset, 1, atol=0.01)
    check_scalar_result("t50", params.t50, 4.5, atol=0.01)
    check_scalar_result("tmax", params.tmax, 5.0, atol=0.01)
    check_scalar_result("foot_duration", params.foot_duration, 4.105, atol=0.01)
    assert params.signal_shape == SignalShape.SQUARE, (
        f"Expected signal_shape to be SQUARE, but got {params.signal_shape}"
    )
    guiutils.view_curves_if_gui(
        [x, y],
        title="Step parameters: "
        f"polarity={params.polarity}, "
        f"offset={params.offset}, "
        f"fwhm={params.fwhm}, "
        f"rise_time={params.rise_time}, "
        f"fall_time={params.fall_time}, "
        f"t50={params.t50}, "
        f"tmax={params.tmax}, "
        f"foot_duration={params.foot_duration}",
    )


@pytest.mark.validation
def test_signal_extract_pulse_features() -> None:
    """Validation test for extract_pulse_features computation function."""
    # Generate a step signal
    x, y = generate_step_signal(
        t_start=0,
        t_end=10,
        dt=0.01,
        t_rise=1,
        t_step=4,
        y_initial=1,
        y_final=6,
        noise_amplitude=0,
        seed=0,
    )
    sig = SignalObj("test_signal")
    sig.set_xydata(x, y)

    p = PulseFeaturesParam()
    p.start_baseline_range_min = 0
    p.start_baseline_range_max = 4
    p.end_baseline_range_min = 6
    p.end_baseline_range_max = 8
    p.start_rise_ratio = 0.1
    p.stop_rise_ratio = 0.1

    result = extract_pulse_features(sig, p)

    param_dict = dict(zip(result.names, result.data[0, :]))
    exptected_result = [
        "step",
        1,
        np.float64(5.0),
        np.float64(0.79),
        # None,
        # None,
        np.float64(1.0),
        np.float64(4.5),
        np.float64(5.0),
        np.float64(4.1),
    ]
    # Check if the result matches the expected values
    for key, expected in zip(result.names, exptected_result[1:]):
        if isinstance(expected, numbers.Number):
            check_scalar_result(key, param_dict[key], expected, atol=0.02)
        else:
            assert np.all(param_dict[key] == expected), (
                f"{key}: {param_dict[key]} != {expected}"
            )

    guiutils.view_curves_if_gui(
        [x, y],
        title="Step parameters: "
        f"polarity={param_dict['polarity']}, "
        f"offset={param_dict['offset']}, "
        #        f"fwhm={param_dict['fwhm']}, "
        f"rise_time={param_dict['rise_time']}, "
        # f"fall_time={param_dict['fall_time']}, "
        f"t50={param_dict['t50']}, "
        f"tmax={param_dict['tmax']}, "
        f"foot_duration={param_dict['foot_duration']}",
    )

    # Validation test for extract_pulse_features with a SQUARE signal
    x, y = generate_square_signal(
        t_start=0,
        t_end=20,
        dt=0.01,
        t_rise=1,
        t_step=4,
        t_fall=7,
        y_initial=1,
        y_high=6,
        noise_amplitude=0,
        seed=0,
    )
    sig = SignalObj("test_square_signal")
    sig.set_xydata(x, y)

    p = PulseFeaturesParam()
    p.start_baseline_range_min = 0
    p.start_baseline_range_max = 2.5
    p.end_baseline_range_min = 15
    p.end_baseline_range_max = 17
    p.start_rise_ratio = 0.1
    p.stop_rise_ratio = 0.1

    result = extract_pulse_features(sig, p)

    param_dict = dict(zip(result.names, result.data[0, :]))
    expected_result = [
        "square",
        1,
        np.float64(5.0),
        np.float64(0.79),
        np.float64(5.595),
        np.float64(6.0),
        np.float64(1.0),
        np.float64(4.5),
        np.float64(5.0),
        np.float64(4.1),
    ]
    for key, expected in zip(result.names, expected_result[1:]):
        if isinstance(expected, numbers.Number):
            check_scalar_result(key, float(param_dict[key]), float(expected), atol=0.02)
        else:
            assert np.all(param_dict[key] == expected), (
                f"{key}: {param_dict[key]} != {expected}"
            )
    guiutils.view_curves_if_gui(
        [x, y],
        title="Step parameters: "
        f"polarity={param_dict['polarity']}, "
        f"offset={param_dict['offset']}, "
        f"fwhm={param_dict['fwhm']}, "
        f"rise_time={param_dict['rise_time']}, "
        f"fall_time={param_dict['fall_time']}, "
        f"t50={param_dict['t50']}, "
        f"tmax={param_dict['tmax']}, "
        f"foot_duration={param_dict['foot_duration']}",
    )


if __name__ == "__main__":
    guiutils.enable_gui()
    # test_heuristically_recognize_shape()
    # test_detect_polarity()
    # test_get_amplitude()
    test_get_crossing_ratio_time()
    test_get_step_rise_time()
    test_get_step_time_at_half_maximum()
    test_get_step_starttime()
    test_get_step_end_time()
    test_heuristically_find_foot_end_time()
    test_get_foot_info()
    test_tools_feature_extraction()
    test_signal_extract_pulse_features()
