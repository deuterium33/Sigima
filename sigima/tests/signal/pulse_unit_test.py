# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for the `sigima.tools.signal.pulse` module.
"""

from __future__ import annotations

import numbers
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pytest

from sigima.enums import SignalShape
from sigima.objects.signal import create_signal
from sigima.proc.signal import PulseFeaturesParam, extract_pulse_features
from sigima.tests import guiutils
from sigima.tests.helpers import check_scalar_result
from sigima.tools.signal import filtering, pulse


@dataclass
class StepSignal:
    """Parameters for generating step signals."""

    y_initial: float = 1.0
    y_final: float = 6.0
    t_start: float = 0.0
    t_end: float = 10.0
    t_rise: float = 1.0
    t_step: float = 4.0
    dt: float = 0.01
    noise_amplitude: float = 0.2
    seed: int = 0


@dataclass
class SquareSignal:
    """Parameters for generating square signals."""

    y_initial: float = 1.0
    y_high: float = 6.0
    t_start: float = 0.0
    t_end: float = 20.0
    t_rise: float = 1.0
    t_step: float = 4.0
    t_fall: float = 7.0
    dt: float = 0.01
    noise_amplitude: float = 0.2
    seed: int = 0


def generate_step_signal(
    t_start: float = 0,
    t_end: float = 10,
    dt: float = 0.01,
    t_rise: float = 2,
    t_step: float = 3,
    y_initial: float = 0,
    y_final: float = 5,
    noise_amplitude: float = 0.2,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a noisy step signal with a linear rise.

    The function creates a time vector and generates a signal that starts at
    `y_initial`, rises linearly to `y_final` starting at `t_step` over a duration
    of `t_rise`, and remains at `y_final` afterwards.
    Gaussian noise is added to the signal.

    Args:
        t_start: Start time of the signal. Defaults to 0.
        t_end: End time of the signal. Defaults to 10.
        dt: Time step for the time vector. Defaults to 0.01.
        t_rise: Duration of the linear rise from `y_initial` to `y_final`.
         Defaults to 2.
        t_step: Time at which the step (rise) begins. Defaults to 3.
        y_initial: Initial value of the signal before the step. Defaults to 0.
        y_final: Final value of the signal after the rise. Defaults to 5.
        noise_amplitude: Standard deviation of the Gaussian noise added to the
         signal. Defaults to 0.2.
        seed: Seed for the random number generator for reproducibility.
         Defaults to None.

    Returns:
        Tuple containing the time vector and noisy step signal.
    """
    # time vector
    x = np.arange(t_start, t_end + dt, dt)

    # creating the signal
    y = np.piecewise(
        x,
        [x < t_step, (x >= t_step) & (x < t_step + t_rise), x >= t_step + t_rise],
        [
            y_initial,
            lambda t: y_initial + (y_final - y_initial) * (t - t_step) / t_rise,
            y_final,
        ],
    )
    rdg = np.random.default_rng(seed)
    noise = rdg.normal(0, noise_amplitude, size=len(y))
    y_noisy = y + noise

    return x, y_noisy


def generate_square_signal(
    t_start: float = 0,
    t_end: float = 15,
    dt: float = 0.01,
    t_rise: float = 2,
    t_step: float = 3,
    square_duration: float = 2,
    t_fall: float = 5,
    y_initial: float = 0,
    y_high: float = 5,
    noise_amplitude: float = 0.2,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic square-like signal with configurable parameters.

    Generates a synthetic square-like signal with configurable rise, plateau, and fall
    times, and adds Gaussian noise.

    Args:
        t_start: Start time of the signal. Defaults to 0.
        t_end: End time of the signal. Defaults to 15.
        dt: Time step for the signal. Defaults to 0.01.
        t_rise: Duration of the rising edge. Defaults to 2.
        t_step: Time at which the rising edge starts. Defaults to 3.
        square_duration: Duration of the high (plateau) part of the signal.
         Defaults to 2.
        t_fall: Duration of the falling edge. Defaults to 5.
        y_initial: Initial (and final) value of the signal. Defaults to 0.
        y_high: Value of the signal during the plateau. Defaults to 5.
        noise_amplitude: Standard deviation of the Gaussian noise added to the
         signal. Defaults to 0.2.
        seed: Seed for the random number generator. Defaults to None.

    Returns:
        Tuple containing the time vector and noisy square signal.
    """
    # time vector
    x = np.arange(t_start, t_end + dt, dt)

    t_start_fall = t_step + t_rise + square_duration
    # creating the signal
    y = np.piecewise(
        x,
        [
            x < t_step,
            (x >= t_step) & (x < t_step + t_rise),
            (x >= t_step + t_rise) & (x < t_start_fall),
            (x >= t_start_fall) & (x < t_fall + t_start_fall),
            x >= t_fall + t_start_fall,
        ],
        [
            y_initial,
            lambda t: y_initial + (y_high - y_initial) * (t - t_step) / t_rise,
            y_high,
            lambda t: y_high - (y_high - y_initial) * (t - t_start_fall) / t_fall,
            y_initial,
        ],
    )
    rdg = np.random.default_rng(seed)
    noise = rdg.normal(0, noise_amplitude, size=len(y))
    y_noisy = y + noise

    return x, y_noisy


@dataclass
class AnalysisParams:
    """Parameters for pulse analysis."""

    start_ratio: float = 0.1
    stop_ratio: float = 0.9
    start_range: tuple[float, float] = (0.0, 4.0)
    end_range: tuple[float, float] = (6.0, 8.0)


@dataclass
class ExpectedFeatures:
    """Expected pulse feature values for validation."""

    polarity: int
    amplitude: float
    rise_time: float
    offset: float
    t50: float
    tmax: float
    foot_duration: float
    fall_time: float | None = None
    fwhm: float | None = None


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
        stop_rise_ratio: Stopping amplitude ratio (e.g., 0.8 for 80%)

    Returns:
        Theoretical rise time between the two ratios
    """
    return (stop_rise_ratio - start_rise_ratio) * t_rise


def theoretical_square_crossing_time(
    t_step: float,
    t_rise: float,
    square_duration: float,
    t_fall: float,
    ratio: float,
    edge: Literal["rise", "fall"] = "rise",
) -> float:
    """Calculate theoretical crossing time for square signal.

    Args:
        t_step: Time when the rise starts
        t_rise: Duration of the rising edge
        square_duration: Duration of the plateau
        t_fall: Duration of the falling edge
        ratio: Crossing ratio (0.0 to 1.0)
        edge: Which edge to calculate

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
    signal_type: Literal["step", "square"],
    expected_shape: SignalShape,
    y_initial: float,
    y_final_or_high: float,
    start_range: tuple[float, float] | None = None,
    end_range: tuple[float, float] | None = None,
) -> None:
    """Helper function to test shape recognition for different signal configurations.

    Args:
        signal_type: Signal shape type
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
    guiutils.view_curves_if_gui([[x, y_noisy]], title=f"{title}: {shape}")

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
    signal_type: Literal["step", "square"],
    polarity_desc: str,
    expected_polarity: int,
    y_initial: float,
    y_final_or_high: float,
    start_range: tuple[float, float] | None = None,
    end_range: tuple[float, float] | None = None,
) -> None:
    """Helper function to test polarity detection for different signal configurations.

    Args:
        signal_type: Signal shape type
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
    guiutils.view_curves_if_gui([[x, y_noisy]], title=f"{title}: {polarity}")

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


def view_baseline_plateau_and_curve(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    signal_type: Literal["step", "square"],
    start_range: tuple[float, float],
    end_range: tuple[float, float],
    plateau_range: tuple[float, float] | None = None,
    other_items: list | None = None,
) -> None:
    """Helper function to visualize signal with baselines and plateau.

    Args:
        x: X data.
        y: Y data.
        title: Title for the plot.
        signal_type: Signal shape type
        start_range: Start baseline range.
        end_range: End baseline range.
        plateau_range: Plateau range for square signals (optional).
        other_items: Additional items to display (optional).
    """
    # pylint: disable=import-outside-toplevel
    from plotpy.builder import make

    from sigima.tests import vistools

    ys = pulse.get_range_mean_y(x, y, start_range)
    ye = pulse.get_range_mean_y(x, y, end_range)
    xs0, xs1 = start_range
    xe0, xe1 = end_range
    items = [
        make.mcurve(x, y, label="Noisy signal"),
        vistools.create_signal_segment(xs0, ys, xs1, ys, "Start baseline"),
        vistools.create_signal_segment(xe0, ye, xe1, ye, "End baseline"),
    ]
    if signal_type == "square":
        if plateau_range is None:
            polarity = pulse.detect_polarity(x, y, start_range, end_range)
            plateau_range = pulse.get_plateau_range(x, y, polarity)
        xp0, xp1 = plateau_range
        yp = pulse.get_range_mean_y(x, y, plateau_range)
        items.append(vistools.create_signal_segment(xp0, yp, xp1, yp, "Plateau"))
    if other_items is not None:
        items.extend(other_items)

    vistools.view_curve_items(items, title=title)


def _test_amplitude_case(
    signal_type: Literal["step", "square"],
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
        signal_type: Signal shape type
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
            view_baseline_plateau_and_curve(
                x,
                y_noisy,
                f"{title}: {amp:.3f}",
                signal_type,
                start_range,
                end_range,
                plateau_range,
            )

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


def _test_crossing_ratio_time_case(
    signal_type: Literal["step", "square"],
    polarity_desc: str,
    y_initial: float,
    y_final_or_high: float,
    start_range: tuple[float, float],
    end_range: tuple[float, float],
    ratio: float,
    edge: Literal["rise", "fall"] = "rise",
    atol: float = 0.1,
    rtol: float = 0.1,
) -> None:
    """Helper function to test crossing ratio time for different signal configurations.

    Args:
        signal_type: Signal shape type
        polarity_desc: Description of polarity ("positive" or "negative")
        y_initial: Initial signal value
        y_final_or_high: Final value (step) or high value (square)
        start_range: Start baseline range for crossing time calculation
        end_range: End baseline range for crossing time calculation
        ratio: Crossing ratio (0.0 to 1.0)
        edge: Which edge to calculate for square signals
        atol: Absolute tolerance for crossing time comparison
        rtol: Relative tolerance for auto-detection comparison
    """
    # Generate signal and calculate expected crossing time
    t_step, t_rise = 3.0, 2.0  # Default signal generation parameters
    if signal_type == "step":
        x, y_noisy = generate_step_signal(
            y_initial=y_initial,
            y_final=y_final_or_high,
            t_step=t_step,
            t_rise=t_rise,
            seed=0,
        )
        expected_ct = theoretical_crossing_time(t_step, t_rise, ratio)
    else:  # square
        square_duration = 2.0  # Default from generate_square_signal
        t_fall = 5.0  # Default duration from generate_square_signal
        x, y_noisy = generate_square_signal(
            y_initial=y_initial,
            y_high=y_final_or_high,
            square_duration=square_duration,
            t_fall=t_fall,
            t_step=t_step,
            t_rise=t_rise,
            seed=0,
        )
        expected_ct = theoretical_square_crossing_time(
            t_step, t_rise, square_duration, t_fall, ratio, edge
        )

    # Create title
    title = (
        f"{signal_type.capitalize()}, {polarity_desc} polarity | "
        f"Get crossing time at {ratio:.1%}"
    )
    if signal_type == "square":
        title += f" ({edge} edge)"

    # Using the same denoise algorithm as in `extract_pulse_features`
    y_noisy = filtering.denoise_preserve_shape(y_noisy)[0]

    # Test with explicit ranges
    ct = pulse.find_crossing_at_ratio(x, y_noisy, ratio, start_range, end_range)
    check_scalar_result(title, ct, expected_ct, atol=atol)

    with guiutils.lazy_qt_app_context() as qt_app:
        if qt_app is not None:
            # pylint: disable=import-outside-toplevel
            from sigima.tests import vistools

            # polarity = pulse.detect_polarity(x, y_noisy, start_range, end_range)
            # plateau_range = pulse.get_plateau_range(x, y_noisy, polarity)
            view_baseline_plateau_and_curve(
                x,
                y_noisy,
                f"{title}: {ct:.3f}",
                signal_type,
                start_range,
                end_range,
                plateau_range=None,
                other_items=[
                    vistools.create_cursor("v", ct, f"Crossing at {ratio:.1%}")
                ],
            )

    # Test auto-detection
    ct_auto = pulse.find_crossing_at_ratio(x, y_noisy, ratio)
    check_scalar_result(f"{title} (auto)", ct_auto, expected_ct, rtol=rtol, atol=atol)


@pytest.mark.parametrize("ratio", [0.2, 0.5, 0.8])
def test_get_crossing_ratio_time(ratio: float) -> None:
    """Unit test for the `pulse.find_crossing_at_ratio` function.

    This test verifies the correct calculation of the crossing time at a given ratio
    for both positive and negative polarity step signals using theoretical calculations
    based on the signal generation parameters.

    Test cases:
        - Step signal with positive polarity.
        - Step signal with negative polarity.
    """
    tcrtc = _test_crossing_ratio_time_case

    tcrtc("step", "positive", 0.0, 5.0, (0.0, 2.0), (6.0, 8.0), ratio)
    tcrtc("step", "negative", 5.0, 2.0, (0.0, 2.0), (6.0, 8.0), ratio)


def _test_step_rise_time_case(
    signal_type: Literal["step", "square"],
    polarity_desc: Literal["positive", "negative"],
    y_initial: float,
    y_final_or_high: float,
    start_range: tuple[float, float],
    end_range: tuple[float, float],
    start_rise_ratio: float,
    stop_rise_ratio: float,
    noise_amplitude: float = 0.1,
    atol: float = 0.1,
    rtol: float = 0.1,
) -> None:
    """Helper function to test step rise time for different signal configurations.

    Args:
        signal_type: Signal shape type
        polarity_desc: Description of polarity
        y_initial: Initial signal value
        y_final_or_high: Final value (step) or high value (square)
        start_range: Start baseline range for rise time calculation
        end_range: End baseline range for rise time calculation
        start_rise_ratio: Starting amplitude ratio for rise time measurement
        stop_rise_ratio: Stopping amplitude ratio (e.g., 0.8 for 80%)
        noise_amplitude: Noise level for signal generation
        atol: Absolute tolerance for rise time comparison
        rtol: Relative tolerance for auto-detection comparison
    """
    # Generate signal and calculate expected rise time
    t_step, t_rise = 3.0, 2.0  # Default signal generation parameters
    expected_rise_time = theoretical_rise_time(
        t_rise, start_rise_ratio, stop_rise_ratio
    )
    rise_or_fall = "Rise" if polarity_desc == "positive" else "Fall"

    if noise_amplitude == 0.0:
        atol /= 10.0  # Tighter check for clean signals

    if signal_type == "step":
        x, y_noisy = generate_step_signal(
            y_initial=y_initial,
            y_final=y_final_or_high,
            t_step=t_step,
            t_rise=t_rise,
            noise_amplitude=noise_amplitude,
            seed=0,
        )
    else:  # square
        square_duration = 2.0  # Default from generate_square_signal
        t_fall = 5.0  # Default duration from generate_square_signal
        x, y_noisy = generate_square_signal(
            y_initial=y_initial,
            y_high=y_final_or_high,
            square_duration=square_duration,
            t_fall=t_fall,
            t_step=t_step,
            t_rise=t_rise,
            noise_amplitude=noise_amplitude,
            seed=0,
        )

    # Create title
    noise_desc = "clean" if noise_amplitude == 0 else "noisy"
    title = (
        f"{signal_type.capitalize()}, {polarity_desc} polarity | "
        f"Get {rise_or_fall.lower()} time ({noise_desc})"
    )

    # Test with explicit ranges
    rise_time = pulse.get_step_rise_time(
        x, y_noisy, start_range, end_range, start_rise_ratio, stop_rise_ratio
    )

    # For noisy negative polarity signals, use looser validation
    if noise_amplitude > 0 and polarity_desc == "negative":
        # Validate that it's reasonable (within factor of 3 of expected)
        assert 0.5 * expected_rise_time <= rise_time <= 3.0 * expected_rise_time, (
            f"{rise_or_fall} time {rise_time:.3f} outside reasonable range "
            f"[{0.5 * expected_rise_time:.3f}, "
            f"{3.0 * expected_rise_time:.3f}] for {noise_desc} {polarity_desc} signal"
        )
    else:
        check_scalar_result(title, rise_time, expected_rise_time, atol=atol)

    with guiutils.lazy_qt_app_context() as qt_app:
        if qt_app is not None:
            # pylint: disable=import-outside-toplevel
            from sigima.tests import vistools

            ct1 = pulse.find_crossing_at_ratio(
                x, y_noisy, start_rise_ratio, start_range, end_range
            )
            ct2 = pulse.find_crossing_at_ratio(
                x, y_noisy, stop_rise_ratio, start_range, end_range
            )
            item = vistools.create_range(
                "h",
                ct1,
                ct2,
                f"{rise_or_fall} time {start_rise_ratio:.0%}-"
                f"{stop_rise_ratio:.0%} = {rise_time:.3f}",
            )

            view_baseline_plateau_and_curve(
                x,
                y_noisy,
                f"{title}: {rise_time:.3f}",
                signal_type,
                start_range,
                end_range,
                plateau_range=None,
                other_items=[item],
            )

    # Test auto-detection
    rise_time_auto = pulse.get_step_rise_time(
        x, y_noisy, start_rise_ratio=start_rise_ratio, stop_rise_ratio=stop_rise_ratio
    )
    check_scalar_result(
        f"{title} (auto)", rise_time_auto, expected_rise_time, rtol=rtol
    )


@pytest.mark.parametrize("noise_amplitude", [0.1, 0.0])
def test_get_step_rise_time(noise_amplitude: float) -> None:
    """Unit test for the `pulse.get_step_rise_time` function.

    This test verifies the correct calculation of the rise time for step signals with
    both positive and negative polarity using theoretical calculations based on
    signal generation parameters.

    Test cases (including noisy and clean signals):
        - Step signal with positive polarity (20%-80% rise time).
        - Step signal with negative polarity (20%-80% rise time).
    """
    tsrtc = _test_step_rise_time_case
    # Standard 20%-80% rise time parameters
    start_ratio, stop_ratio = 0.2, 0.8

    # Step signals with positive polarity
    na = noise_amplitude
    tsrtc("step", "positive", 0.0, 5.0, (0, 2), (6, 8), start_ratio, stop_ratio, na)
    tsrtc("step", "negative", 5.0, 2.0, (0, 2), (6, 8), start_ratio, stop_ratio, na)


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
    guiutils.view_curves_if_gui([[x, y]], title=f"Foot end time = {time:.3f}")


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

    guiutils.view_curves_if_gui(
        [[x, y]],
        title=f"Foot info: x_end={foot_info.x_end:.3f},"
        f"threshold={foot_info.threshold:.3f}",
    )


def view_pulse_features(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    signal_type: Literal["step", "square"],
    params: pulse.PulseFeatures,
    start_range: tuple[float, float],
    end_range: tuple[float, float],
) -> None:
    """Helper function to visualize pulse features.

    Args:
        x: X data.
        y: Y data.
        title: Title for the plot.
        signal_type: Signal shape type
        params: Extracted pulse features.
        start_range: Start baseline range.
        end_range: End baseline range.
    """
    # pylint: disable=import-outside-toplevel
    from sigima.tests import vistools

    params_text = "<br>".join(
        [
            f"<b>Extracted {signal_type} parameters:</b>",
            f"Polarity: {params.polarity}",
            f"Amplitude: {params.amplitude}",
            f"Rise time: {params.rise_time}",
            f"Fall time: {params.fall_time}",
            f"FWHM: {params.fwhm}",
            f"Offset: {params.offset}",
            f"T50: {params.t50}",
            f"Tmax: {params.tmax}",
            f"Foot duration: {params.foot_duration}",
        ]
    )
    view_baseline_plateau_and_curve(
        x,
        y,
        title,
        signal_type,
        start_range,
        end_range,
        plateau_range=None,
        other_items=[vistools.create_label(params_text)],
    )


def _extract_and_validate_step_features(
    x: np.ndarray,
    y: np.ndarray,
    analysis: AnalysisParams,
    expected: ExpectedFeatures,
) -> pulse.PulseFeatures:
    """Helper function to extract and validate step signal features.

    Args:
        x: X data array
        y: Y data array
        analysis: Analysis parameters for pulse feature extraction
        expected: Expected feature values for validation

    Returns:
        Extracted pulse features
    """
    # Extract features while ignoring FWHM warnings for noisy signals
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        params = pulse.extract_pulse_features(
            x,
            y,
            analysis.start_range,
            analysis.end_range,
            analysis.start_ratio,
            analysis.stop_ratio,
        )

    # Visualize results if GUI is available
    with guiutils.lazy_qt_app_context() as qt_app:
        if qt_app is not None:
            view_pulse_features(
                x,
                y,
                "Step signal feature extraction",
                "step",
                params,
                analysis.start_range,
                analysis.end_range,
            )

    # Validate that we got the correct type
    assert isinstance(params, pulse.PulseFeatures), (
        f"Expected PulseFeatures, got {type(params)}"
    )

    # Validate signal shape
    assert params.signal_shape == SignalShape.STEP, (
        f"Expected signal_shape to be STEP, but got {params.signal_shape}"
    )

    # Validate numerical features
    check_scalar_result(
        "[step] polarity", params.polarity, expected.polarity, atol=1e-8
    )
    check_scalar_result(
        "[step] amplitude", params.amplitude, expected.amplitude, atol=0.01
    )
    check_scalar_result(
        "[step] rise_time", params.rise_time, expected.rise_time, atol=0.01
    )
    check_scalar_result("[step] offset", params.offset, expected.offset, atol=0.01)
    check_scalar_result("[step] t50", params.t50, expected.t50, atol=0.05)
    check_scalar_result("[step] tmax", params.tmax, expected.tmax, atol=0.5)
    check_scalar_result(
        "[step] foot_duration", params.foot_duration, expected.foot_duration, atol=0.1
    )

    # Validate that step-specific features are None
    assert params.fall_time is None, (
        f"Expected fall_time to be None for step signal, but got {params.fall_time}"
    )
    assert params.fwhm is None, (
        f"Expected fwhm to be None for step signal, but got {params.fwhm}"
    )

    return params


def _extract_and_validate_square_features(
    x: np.ndarray,
    y: np.ndarray,
    analysis: AnalysisParams,
    expected: ExpectedFeatures,
) -> pulse.PulseFeatures:
    """Helper function to extract and validate square signal features.

    Args:
        x: X data array
        y: Y data array
        analysis: Analysis parameters for pulse feature extraction
        expected: Expected feature values for validation

    Returns:
        Extracted pulse features
    """
    # Extract features while ignoring FWHM warnings for noisy signals
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        params = pulse.extract_pulse_features(
            x,
            y,
            analysis.start_range,
            analysis.end_range,
            analysis.start_ratio,
            analysis.stop_ratio,
        )

    # Visualize results if GUI is available
    with guiutils.lazy_qt_app_context() as qt_app:
        if qt_app is not None:
            view_pulse_features(
                x,
                y,
                "Square signal feature extraction",
                "square",
                params,
                analysis.start_range,
                analysis.end_range,
            )

    # Validate that we got the correct type
    assert isinstance(params, pulse.PulseFeatures), (
        f"Expected PulseFeatures, got {type(params)}"
    )

    # Validate signal shape
    assert params.signal_shape == SignalShape.SQUARE, (
        f"Expected signal_shape to be SQUARE, but got {params.signal_shape}"
    )

    # Validate numerical features
    check_scalar_result(
        "[square] polarity", params.polarity, expected.polarity, atol=1e-8
    )
    check_scalar_result(
        "[square] amplitude", params.amplitude, expected.amplitude, atol=0.05
    )
    check_scalar_result(
        "[square] rise_time", params.rise_time, expected.rise_time, atol=0.1
    )
    check_scalar_result(
        "[square] fall_time", params.fall_time, expected.fall_time, atol=0.3
    )
    check_scalar_result("[square] fwhm", params.fwhm, expected.fwhm, atol=0.5)
    check_scalar_result("[square] offset", params.offset, expected.offset, atol=0.01)
    check_scalar_result("[square] t50", params.t50, expected.t50, atol=0.01)
    check_scalar_result("[square] tmax", params.tmax, expected.tmax, atol=0.5)
    check_scalar_result(
        "[square] foot_duration", params.foot_duration, expected.foot_duration, atol=0.1
    )

    return params


def test_step_feature_extraction() -> None:
    """Test feature extraction for step signals.

    Validates that pulse feature extraction correctly identifies and measures
    all relevant parameters for a step signal, including polarity, amplitude,
    rise time, timing features, and baseline characteristics.
    """
    # Define signal parameters
    signal_params = StepSignal()

    # Define analysis parameters
    analysis = AnalysisParams()

    # Calculate expected values
    expected = ExpectedFeatures(
        polarity=1,
        amplitude=theoretical_step_amplitude(
            signal_params.y_initial, signal_params.y_final
        ),
        rise_time=theoretical_rise_time(
            signal_params.t_rise, analysis.start_ratio, analysis.stop_ratio
        ),
        offset=signal_params.y_initial,
        t50=signal_params.t_start + signal_params.t_step + 0.5 * signal_params.t_rise,
        tmax=signal_params.t_start + signal_params.t_step + signal_params.t_rise,
        foot_duration=signal_params.t_step - signal_params.t_start,
    )

    # Generate test signal
    x, y = generate_step_signal(
        y_initial=signal_params.y_initial,
        y_final=signal_params.y_final,
        t_start=signal_params.t_start,
        t_end=signal_params.t_end,
        t_rise=signal_params.t_rise,
        t_step=signal_params.t_step,
        dt=signal_params.dt,
        noise_amplitude=signal_params.noise_amplitude,
        seed=signal_params.seed,
    )

    # Extract and validate features
    _extract_and_validate_step_features(x, y, analysis, expected)


def test_square_feature_extraction() -> None:
    """Test feature extraction for square signals.

    Validates that pulse feature extraction correctly identifies and measures
    all relevant parameters for a square signal, including polarity, amplitude,
    rise/fall times, FWHM, timing features, and baseline characteristics.
    """
    # Define signal parameters with custom ranges for square signal
    signal_params = SquareSignal()

    # Define analysis parameters with custom ranges for square signal
    analysis = AnalysisParams(
        start_range=(0.0, 2.5),
        end_range=(15.0, 17.0),
    )

    # Calculate expected values
    expected = ExpectedFeatures(
        polarity=1,
        amplitude=theoretical_square_amplitude(
            signal_params.y_initial, signal_params.y_high
        ),
        rise_time=theoretical_rise_time(
            signal_params.t_rise, analysis.start_ratio, analysis.stop_ratio
        ),
        fall_time=theoretical_rise_time(
            signal_params.t_fall, analysis.start_ratio, analysis.stop_ratio
        ),
        fwhm=signal_params.t_fall - signal_params.t_rise,  # 7 - 1 = 6
        offset=signal_params.y_initial,
        t50=signal_params.t_start + signal_params.t_step + 0.5 * signal_params.t_rise,
        tmax=signal_params.t_start + signal_params.t_step + signal_params.t_rise,
        foot_duration=signal_params.t_step - signal_params.t_start,
    )

    # Generate test signal
    x, y = generate_square_signal(
        y_initial=signal_params.y_initial,
        y_high=signal_params.y_high,
        t_start=signal_params.t_start,
        t_end=signal_params.t_end,
        t_rise=signal_params.t_rise,
        t_step=signal_params.t_step,
        t_fall=signal_params.t_fall,
        dt=signal_params.dt,
        noise_amplitude=signal_params.noise_amplitude,
        seed=signal_params.seed,
    )

    # Extract and validate features
    _extract_and_validate_square_features(x, y, analysis, expected)


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
    sig = create_signal("Test Signal", x, y)

    p = PulseFeaturesParam()
    p.start_range_min = 0
    p.start_range_max = 4
    p.end_range_min = 6
    p.end_range_max = 8
    p.start_rise_ratio = 0.1
    p.stop_rise_ratio = 0.9

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
        sig,
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
    sig = create_signal("Test Square Signal", x, y)

    p = PulseFeaturesParam()
    p.start_range_min = 0
    p.start_range_max = 2.5
    p.end_range_min = 15
    p.end_range_max = 17
    p.start_rise_ratio = 0.1
    p.stop_rise_ratio = 0.9

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
        sig,
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
    # test_get_crossing_ratio_time(0.2)
    # test_get_crossing_ratio_time(0.5)
    # test_get_crossing_ratio_time(0.8)
    # test_get_step_rise_time(0.1)
    # test_get_step_rise_time(0.0)
    # test_heuristically_find_foot_end_time()
    test_get_foot_info()
    test_step_feature_extraction()
    test_square_feature_extraction()
    test_signal_extract_pulse_features()
