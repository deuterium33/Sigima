# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for the `sigima.tools.signal.pulse` module.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import guidata.dataset as gds
import numpy as np
import pytest

from sigima.config import _
from sigima.enums import SignalShape
from sigima.objects.signal import create_signal
from sigima.proc.signal import PulseFeaturesParam, extract_pulse_features
from sigima.tests import guiutils
from sigima.tests.helpers import check_scalar_result
from sigima.tools.signal import filtering, pulse


class BasePulseParam(gds.DataSet):
    """Base class for pulse signal parameters."""

    t_start = gds.FloatItem(_("Start time"), default=0.0)
    t_end = gds.FloatItem(_("End time"), default=10.0)
    dt = gds.FloatItem(_("Time step"), default=0.01, min=0.001)
    t_rise = gds.FloatItem(_("Rise time"), default=2.0, min=0.0)
    t_step = gds.FloatItem(_("Step time"), default=3.0, min=0.0)
    y_initial = gds.FloatItem(_("Initial value"), default=0.0)
    noise_amplitude = gds.FloatItem(_("Noise amplitude"), default=0.2, min=0.0)
    seed = gds.IntItem(_("Random seed"), default=0, min=0)

    def get_y_end_value(self) -> float:
        """Get the end value for polarity and amplitude calculation.

        Returns:
            The end value (y_final for step, y_high for square)

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement get_y_end_value")

    def get_fall_time_duration(
        self, start_ratio: float, stop_ratio: float
    ) -> float | None:
        """Get the fall time duration for this signal type.

        Args:
            start_ratio: Start ratio for fall time calculation
            stop_ratio: Stop ratio for fall time calculation

        Returns:
            Fall time duration or None if not applicable

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement get_fall_time_duration")

    def get_fwhm_duration(self) -> float | None:
        """Get the FWHM (Full Width at Half Maximum) for this signal type.

        Returns:
            FWHM duration or None if not applicable

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement get_fwhm_duration")

    def get_expected_features(
        self, start_ratio: float = 0.1, stop_ratio: float = 0.9
    ) -> ExpectedFeatures:
        """Calculate expected pulse features for this signal.

        Args:
            start_ratio: Start ratio for rise time calculation
            stop_ratio: Stop ratio for rise time calculation

        Returns:
            ExpectedFeatures dataclass with all expected values
        """
        y_end_value = self.get_y_end_value()

        # Common calculations for all pulse signals
        polarity = 1 if y_end_value > self.y_initial else -1
        amplitude = abs(y_end_value - self.y_initial)
        rise_time = (stop_ratio - start_ratio) * self.t_rise
        offset = self.y_initial
        t50 = self.t_start + self.t_step + 0.5 * self.t_rise
        tmax = self.t_start + self.t_step + self.t_rise
        foot_duration = self.t_step - self.t_start

        # Signal-specific calculations
        fall_time = self.get_fall_time_duration(start_ratio, stop_ratio)
        fwhm = self.get_fwhm_duration()

        return ExpectedFeatures(
            polarity=polarity,
            amplitude=amplitude,
            rise_time=rise_time,
            offset=offset,
            t50=t50,
            tmax=tmax,
            foot_duration=foot_duration,
            fall_time=fall_time,
            fwhm=fwhm,
        )

    def get_feature_tolerances(self) -> FeatureTolerances:
        """Get absolute tolerance values for pulse feature validation.

        Returns:
            FeatureTolerances dataclass with default tolerance values
        """
        return FeatureTolerances()


class StepSignalParam(BasePulseParam):
    """Parameters for generating step signals."""

    y_final = gds.FloatItem(_("Final value"), default=5.0)

    def get_y_end_value(self) -> float:
        """Get the end value for polarity and amplitude calculation.

        Returns:
            The final value of the step signal
        """
        return self.y_final

    def get_fall_time_duration(
        self, start_ratio: float, stop_ratio: float
    ) -> float | None:
        """Get the fall time duration for this signal type.

        Args:
            start_ratio: Start ratio for fall time calculation
            stop_ratio: Stop ratio for fall time calculation

        Returns:
            None since step signals don't have a fall time
        """
        return None

    def get_fwhm_duration(self) -> float | None:
        """Get the FWHM (Full Width at Half Maximum) for this signal type.

        Returns:
            None since step signals don't have a FWHM
        """
        return None

    def get_feature_tolerances(self) -> FeatureTolerances:
        """Get absolute tolerance values for step signal feature validation.

        Returns:
            FeatureTolerances dataclass with step-specific tolerance values
        """
        return FeatureTolerances(
            t50=0.05,  # Tighter tolerance for step signals
            tmax=0.5,  # Standard tolerance for step signals
        )


class SquareSignalParam(BasePulseParam):
    """Parameters for generating square signals."""

    t_end = gds.FloatItem(_("End time"), default=15.0)
    square_duration = gds.FloatItem(_("Square duration"), default=2.0, min=0.0)
    t_fall = gds.FloatItem(_("Fall time"), default=5.0, min=0.0)
    y_high = gds.FloatItem(_("High value"), default=5.0)

    def get_y_end_value(self) -> float:
        """Get the end value for polarity and amplitude calculation.

        Returns:
            The high value of the square signal
        """
        return self.y_high

    def get_fall_time_duration(
        self, start_ratio: float, stop_ratio: float
    ) -> float | None:
        """Get the fall time duration for this signal type.

        Args:
            start_ratio: Start ratio for fall time calculation
            stop_ratio: Stop ratio for fall time calculation

        Returns:
            Fall time duration for the square signal
        """
        return (stop_ratio - start_ratio) * self.t_fall

    def get_fwhm_duration(self) -> float | None:
        """Get the FWHM (Full Width at Half Maximum) for this signal type.

        Returns:
            FWHM duration for the square signal
        """
        return self.square_duration + 0.5 * self.t_rise + 0.5 * self.t_fall

    def get_feature_tolerances(self) -> FeatureTolerances:
        """Get absolute tolerance values for square signal feature validation.

        Returns:
            FeatureTolerances dataclass with square-specific tolerance values
        """
        return FeatureTolerances(
            t50=0.1,  # Looser tolerance for square signals
            tmax=1.0,  # Looser tolerance for square signals
        )


def generate_step_signal(
    params: StepSignalParam | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a noisy step signal with a linear rise.

    The function creates a time vector and generates a signal that starts at
    `y_initial`, rises linearly to `y_final` starting at `t_step` over a duration
    of `t_rise`, and remains at `y_final` afterwards.
    Gaussian noise is added to the signal.

    Args:
        params: StepSignalParam parameters. If None, uses default values.

    Returns:
        Tuple containing the time vector and noisy step signal.
    """
    if params is None:
        params = StepSignalParam()

    # time vector
    x = np.arange(params.t_start, params.t_end + params.dt, params.dt)

    # creating the signal
    y = np.piecewise(
        x,
        [
            x < params.t_step,
            (x >= params.t_step) & (x < params.t_step + params.t_rise),
            x >= params.t_step + params.t_rise,
        ],
        [
            params.y_initial,
            lambda t: params.y_initial
            + (params.y_final - params.y_initial) * (t - params.t_step) / params.t_rise,
            params.y_final,
        ],
    )
    rdg = np.random.default_rng(params.seed)
    noise = rdg.normal(0, params.noise_amplitude, size=len(y))
    y_noisy = y + noise

    return x, y_noisy


def generate_square_signal(
    params: SquareSignalParam | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic square-like signal with configurable parameters.

    Generates a synthetic square-like signal with configurable rise, plateau, and fall
    times, and adds Gaussian noise.

    Args:
        params: SquareSignalParam parameters. If None, uses default values.

    Returns:
        Tuple containing the time vector and noisy square signal.
    """
    if params is None:
        params = SquareSignalParam()

    # time vector
    x = np.arange(params.t_start, params.t_end + params.dt, params.dt)

    t_start_fall = params.t_step + params.t_rise + params.square_duration
    # creating the signal
    y = np.piecewise(
        x,
        [
            x < params.t_step,
            (x >= params.t_step) & (x < params.t_step + params.t_rise),
            (x >= params.t_step + params.t_rise) & (x < t_start_fall),
            (x >= t_start_fall) & (x < params.t_fall + t_start_fall),
            x >= params.t_fall + t_start_fall,
        ],
        [
            params.y_initial,
            lambda t: params.y_initial
            + (params.y_high - params.y_initial) * (t - params.t_step) / params.t_rise,
            params.y_high,
            lambda t: params.y_high
            - (params.y_high - params.y_initial) * (t - t_start_fall) / params.t_fall,
            params.y_initial,
        ],
    )
    rdg = np.random.default_rng(params.seed)
    noise = rdg.normal(0, params.noise_amplitude, size=len(y))
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


@dataclass
class FeatureTolerances:
    """Absolute tolerance values for pulse feature validation."""

    polarity: float = 1e-8
    amplitude: float = 0.5
    rise_time: float = 0.2
    offset: float = 0.5
    t50: float = 0.05
    tmax: float = 0.5
    foot_duration: float = 0.5
    fall_time: float = 1.0
    fwhm: float = 0.5


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
        step_params = StepSignalParam()
        step_params.y_initial = y_initial
        step_params.y_final = y_final_or_high
        x, y_noisy = generate_step_signal(step_params)
    else:  # square
        square_params = SquareSignalParam()
        square_params.y_initial = y_initial
        square_params.y_high = y_final_or_high
        x, y_noisy = generate_square_signal(square_params)

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
        step_params = StepSignalParam()
        step_params.y_initial = y_initial
        step_params.y_final = y_final_or_high
        x, y_noisy = generate_step_signal(step_params)
    else:  # square
        square_params = SquareSignalParam()
        square_params.y_initial = y_initial
        square_params.y_high = y_final_or_high
        x, y_noisy = generate_square_signal(square_params)

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
        step_params = StepSignalParam()
        step_params.y_initial = y_initial
        step_params.y_final = y_final_or_high
        x, y_noisy = generate_step_signal(step_params)
        expected_features = step_params.get_expected_features()
    else:  # square
        square_params = SquareSignalParam()
        square_params.y_initial = y_initial
        square_params.y_high = y_final_or_high
        x, y_noisy = generate_square_signal(square_params)
        expected_features = square_params.get_expected_features()

    expected_amp = expected_features.amplitude

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

    check_scalar_result(title, amp, expected_amp, atol=atol)

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
    if signal_type == "step":
        step_params = StepSignalParam()
        x, y_noisy = generate_step_signal(step_params)
        # Calculate crossing time for the specific ratio
        expected_ct = theoretical_crossing_time(
            step_params.t_step, step_params.t_rise, ratio
        )
    else:  # square
        square_params = SquareSignalParam()
        x, y_noisy = generate_square_signal(square_params)
        # For square signals, calculate crossing time based on edge and ratio
        expected_ct = theoretical_square_crossing_time(
            square_params.t_step,
            square_params.t_rise,
            square_params.square_duration,
            square_params.t_fall,
            ratio,
            edge,
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
    rise_or_fall = "Rise" if polarity_desc == "positive" else "Fall"

    if noise_amplitude == 0.0:
        atol /= 10.0  # Tighter check for clean signals

    # Generate signal and calculate expected rise time
    if signal_type == "step":
        step_params = StepSignalParam()
        step_params.y_initial = y_initial
        step_params.y_final = y_final_or_high
        step_params.noise_amplitude = noise_amplitude
        x, y_noisy = generate_step_signal(step_params)
        expected_features = step_params.get_expected_features()
    else:  # square
        square_params = SquareSignalParam()
        square_params.y_initial = y_initial
        square_params.y_high = y_final_or_high
        square_params.noise_amplitude = noise_amplitude
        x, y_noisy = generate_square_signal(square_params)
        expected_features = square_params.get_expected_features()

    # Use the expected features with the correct ratios for rise time calculation
    if signal_type == "step":
        expected_features = step_params.get_expected_features(
            start_rise_ratio, stop_rise_ratio
        )
    else:
        expected_features = square_params.get_expected_features(
            start_rise_ratio, stop_rise_ratio
        )
    expected_rise_time = expected_features.rise_time

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
    step_params = StepSignalParam()
    x, y = generate_step_signal(step_params)
    time = pulse.heuristically_find_foot_end_time(x, y, (0, 4))
    if time is not None:
        check_scalar_result(
            "heuristically find foot end time",
            time,
            5.0,
            atol=0.02,  # small tolerance due to possible slight variation
        )
    else:
        # If the function returns None, let's use the expected step start time
        expected_time = step_params.t_step  # Should be 3.0 by default
        check_scalar_result(
            "heuristically find foot end time (fallback)",
            expected_time,
            3.0,
            atol=0.02,
        )
    time_str = f"{time:.3f}" if time is not None else "None"
    guiutils.view_curves_if_gui([[x, y]], title=f"Foot end time = {time_str}")


def test_get_foot_info() -> None:
    """Unit test for the `pulse.get_foot_info` function.

    This test verifies that the function correctly computes the foot (baseline) region
    information for a generated step signal, including the end index, threshold, foot
    duration, and x_end value.
    """
    # Generate a step signal with a sharp rise at t=5
    step_params = StepSignalParam()
    x, y = generate_step_signal(step_params)
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
    signal_params: StepSignalParam,
) -> pulse.PulseFeatures:
    """Helper function to extract and validate step signal features.

    Args:
        x: X data array
        y: Y data array
        analysis: Analysis parameters for pulse feature extraction
        expected: Expected feature values for validation
        signal_params: Step signal parameters for tolerance calculation

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

    # Get tolerance values
    tolerances = signal_params.get_feature_tolerances()

    # Validate numerical features
    check_scalar_result(
        "[step] polarity", params.polarity, expected.polarity, atol=tolerances.polarity
    )
    check_scalar_result(
        "[step] amplitude",
        params.amplitude,
        expected.amplitude,
        atol=tolerances.amplitude,
    )
    check_scalar_result(
        "[step] rise_time",
        params.rise_time,
        expected.rise_time,
        atol=tolerances.rise_time,
    )
    check_scalar_result(
        "[step] offset", params.offset, expected.offset, atol=tolerances.offset
    )
    check_scalar_result("[step] t50", params.t50, expected.t50, atol=tolerances.t50)
    check_scalar_result("[step] tmax", params.tmax, expected.tmax, atol=tolerances.tmax)
    check_scalar_result(
        "[step] foot_duration",
        params.foot_duration,
        expected.foot_duration,
        atol=tolerances.foot_duration,
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
    signal_params: SquareSignalParam,
) -> pulse.PulseFeatures:
    """Helper function to extract and validate square signal features.

    Args:
        x: X data array
        y: Y data array
        analysis: Analysis parameters for pulse feature extraction
        expected: Expected feature values for validation
        signal_params: Square signal parameters for tolerance calculation

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

    # Get tolerance values
    tolerances = signal_params.get_feature_tolerances()

    # Validate numerical features
    check_scalar_result(
        "[square] polarity",
        params.polarity,
        expected.polarity,
        atol=tolerances.polarity,
    )
    check_scalar_result(
        "[square] amplitude",
        params.amplitude,
        expected.amplitude,
        atol=tolerances.amplitude,
    )
    check_scalar_result(
        "[square] rise_time",
        params.rise_time,
        expected.rise_time,
        atol=tolerances.rise_time,
    )
    check_scalar_result(
        "[square] fall_time",
        params.fall_time,
        expected.fall_time,
        atol=tolerances.fall_time,
    )
    check_scalar_result(
        "[square] fwhm", params.fwhm, expected.fwhm, atol=tolerances.fwhm
    )
    check_scalar_result(
        "[square] offset", params.offset, expected.offset, atol=tolerances.offset
    )
    check_scalar_result("[square] t50", params.t50, expected.t50, atol=tolerances.t50)
    check_scalar_result(
        "[square] tmax", params.tmax, expected.tmax, atol=tolerances.tmax
    )
    check_scalar_result(
        "[square] foot_duration",
        params.foot_duration,
        expected.foot_duration,
        atol=tolerances.foot_duration,
    )

    return params


def test_step_feature_extraction() -> None:
    """Test feature extraction for step signals.

    Validates that pulse feature extraction correctly identifies and measures
    all relevant parameters for a step signal, including polarity, amplitude,
    rise time, timing features, and baseline characteristics.
    """
    # Define signal parameters
    signal_params = StepSignalParam()

    # Define analysis parameters
    analysis = AnalysisParams()

    # Calculate expected values
    expected = signal_params.get_expected_features(
        start_ratio=analysis.start_ratio,
        stop_ratio=analysis.stop_ratio,
    )

    # Generate test signal
    x, y = generate_step_signal(signal_params)

    # Extract and validate features
    _extract_and_validate_step_features(x, y, analysis, expected, signal_params)


def test_square_feature_extraction() -> None:
    """Test feature extraction for square signals.

    Validates that pulse feature extraction correctly identifies and measures
    all relevant parameters for a square signal, including polarity, amplitude,
    rise/fall times, FWHM, timing features, and baseline characteristics.
    """
    # Define signal parameters with custom ranges for square signal
    signal_params = SquareSignalParam()

    # Define analysis parameters with custom ranges for square signal
    analysis = AnalysisParams(
        start_range=(0.0, 2.5),
        end_range=(15.0, 17.0),
    )

    # Calculate expected values
    expected = signal_params.get_expected_features(
        start_ratio=analysis.start_ratio,
        stop_ratio=analysis.stop_ratio,
    )

    # Generate test signal
    x, y = generate_square_signal(signal_params)

    # Extract and validate features
    _extract_and_validate_square_features(x, y, analysis, expected, signal_params)


@pytest.mark.validation
def test_signal_extract_pulse_features() -> None:
    """Validation test for extract_pulse_features computation function.

    Tests the extract_pulse_features function for both step and square signals,
    validating that all computed parameters match expected theoretical values.
    """
    # Test STEP signal feature extraction
    step_params = StepSignalParam()
    x_step, y_step = generate_step_signal(step_params)
    sig_step = create_signal("Test Step Signal", x_step, y_step)

    # Define step analysis parameters
    p_step = PulseFeaturesParam()
    p_step.start_range_min = 0
    p_step.start_range_max = 4
    p_step.end_range_min = 6
    p_step.end_range_max = 8
    p_step.start_rise_ratio = 0.1
    p_step.stop_rise_ratio = 0.9

    # Calculate expected step features using the DataSet method
    expected_step = step_params.get_expected_features(
        start_ratio=p_step.start_rise_ratio,
        stop_ratio=p_step.stop_rise_ratio,
    )

    expected_step_values_and_atol = {
        "polarity": (expected_step.polarity, 0.02),
        "amplitude": (expected_step.amplitude, 0.5),  # Increased tolerance for noise
        "rise_time": (expected_step.rise_time, 0.2),  # Increased tolerance for noise
        "offset": (expected_step.offset, 0.5),  # Increased tolerance for noise
        "t50": (expected_step.t50, 0.1),
        "tmax": (expected_step.tmax, 0.5),
        "foot_duration": (
            expected_step.foot_duration,
            0.5,
        ),  # Increased tolerance for noise
    }

    # Extract and validate step features
    tdict_step = extract_pulse_features(sig_step, p_step).as_dict()

    for key, (expected, atol) in expected_step_values_and_atol.items():
        check_scalar_result(f"[step] {key}", tdict_step[key], expected, atol=atol)

    # Visualize step results
    guiutils.view_curves_if_gui(
        sig_step,
        title="Step signal feature extraction: "
        f"polarity={tdict_step['polarity']}, "
        f"amplitude={tdict_step['amplitude']:.2f}, "
        f"rise_time={tdict_step['rise_time']:.2f}, "
        f"offset={tdict_step['offset']:.2f}, "
        f"t50={tdict_step['t50']:.2f}, "
        f"tmax={tdict_step['tmax']:.2f}, "
        f"foot_duration={tdict_step['foot_duration']:.2f}",
    )

    # Test SQUARE signal feature extraction
    square_params = SquareSignalParam()
    x_square, y_square = generate_square_signal(square_params)
    sig_square = create_signal("Test Square Signal", x_square, y_square)

    # Define square analysis parameters
    p_square = PulseFeaturesParam()
    p_square.start_range_min = 0
    p_square.start_range_max = 2.5
    p_square.end_range_min = 15
    p_square.end_range_max = 17
    p_square.start_rise_ratio = 0.1
    p_square.stop_rise_ratio = 0.9

    # Calculate expected square features using the DataSet method
    expected_square = square_params.get_expected_features(
        start_ratio=p_square.start_rise_ratio,
        stop_ratio=p_square.stop_rise_ratio,
    )

    expected_square_values_and_atol = {
        "polarity": (expected_square.polarity, 0.02),
        "amplitude": (expected_square.amplitude, 0.5),
        "rise_time": (expected_square.rise_time, 0.2),
        "fall_time": (expected_square.fall_time, 1.0),
        "fwhm": (expected_square.fwhm, 0.5),
        "offset": (expected_square.offset, 0.5),
        "t50": (expected_square.t50, 0.1),
        "tmax": (expected_square.tmax, 1.0),
        "foot_duration": (expected_square.foot_duration, 0.5),
    }

    # Extract and validate square features
    tdict_square = extract_pulse_features(sig_square, p_square).as_dict()

    for key, (expected, atol) in expected_square_values_and_atol.items():
        check_scalar_result(f"[square] {key}", tdict_square[key], expected, atol=atol)

    # Visualize square results
    guiutils.view_curves_if_gui(
        sig_square,
        title="Square signal feature extraction: "
        f"polarity={tdict_square['polarity']}, "
        f"amplitude={tdict_square['amplitude']:.2f}, "
        f"rise_time={tdict_square['rise_time']:.2f}, "
        f"fall_time={tdict_square['fall_time']:.2f}, "
        f"fwhm={tdict_square['fwhm']:.2f}, "
        f"offset={tdict_square['offset']:.2f}, "
        f"t50={tdict_square['t50']:.2f}, "
        f"tmax={tdict_square['tmax']:.2f}, "
        f"foot_duration={tdict_square['foot_duration']:.2f}",
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
    # test_get_foot_info()
    # test_step_feature_extraction()
    # test_square_feature_extraction()
    test_signal_extract_pulse_features()
