# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for the `sigima.tools.signal.pulse` module.
"""

import numbers

import numpy as np
import pytest

from sigima.enums import SignalShape
from sigima.objects.signal import create_signal
from sigima.proc.signal import ParametersParam, SignalObj, get_parameters
from sigima.tests import guiutils
from sigima.tests.data import generate_square_signal, generate_step_signal
from sigima.tests.helpers import check_scalar_result
from sigima.tools.signal import pulse


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
    x, y_noisy = generate_step_signal(seed=0)
    shape = pulse.heuristically_recognize_shape(x, y_noisy, (0, 2), (4, 8))
    assert shape == SignalShape.STEP, f"Expected {SignalShape.STEP}, got {shape}"
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(signal, f"Detected shape = {shape}")

    shape = pulse.heuristically_recognize_shape(x, y_noisy)
    assert shape == SignalShape.STEP, f"Expected {SignalShape.STEP}, got {shape}"

    x, y_noisy = generate_square_signal(seed=0)
    shape = pulse.heuristically_recognize_shape(x, y_noisy, (0, 2), (12, 14))
    assert shape == SignalShape.SQUARE, f"Expected {SignalShape.SQUARE}, got {shape}"
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(signal, f"Detected shape = {shape}")

    x, y_noisy = generate_step_signal(seed=0, y_initial=5, y_final=2)
    shape = pulse.heuristically_recognize_shape(x, y_noisy, (0, 2), (4, 8))
    assert shape == SignalShape.STEP, f"Expected {SignalShape.STEP}, got {shape}"
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(signal, f"Detected shape = {shape}")

    x, y_noisy = generate_square_signal(seed=0, y_initial=5, y_high=2)
    shape = pulse.heuristically_recognize_shape(x, y_noisy, (0, 2), (12, 14))
    assert shape == SignalShape.SQUARE, f"Expected {SignalShape.SQUARE}, got {shape}"
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(signal, title=f"Detected shape = {shape}")


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
    x, y_noisy = generate_step_signal(seed=0)
    polarity = pulse.detect_polarity(x, y_noisy, (0, 2), (4, 8))
    check_scalar_result("step, detection positive polarity", polarity, 1)
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(signal, title=f"Detected polarity = {polarity}")

    polarity = pulse.detect_polarity(x, y_noisy)
    check_scalar_result("step, detection positive polarity", polarity, 1)
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(signal, title=f"Detected polarity = {polarity}")

    x, y_noisy = generate_square_signal(seed=0)
    polarity = pulse.detect_polarity(x, y_noisy, (0, 2), (12, 14))
    check_scalar_result("step, detection positive polarity", polarity, 1)
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(signal, title=f"Detected polarity = {polarity}")

    x, y_noisy = generate_step_signal(seed=0, y_initial=5, y_final=2)
    polarity = pulse.detect_polarity(x, y_noisy, (0, 2), (4, 8))
    check_scalar_result("step, detection negative polarity", polarity, -1)
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(signal, title=f"Detected polarity = {polarity}")

    x, y_noisy = generate_square_signal(seed=0, y_initial=5, y_high=2)
    polarity = pulse.detect_polarity(x, y_noisy, (0, 2), (12, 14))
    check_scalar_result("step, detection negative polarity", polarity, -1)
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(signal, title=f"Detected polarity = {polarity}")


def test_get_amplitude() -> None:
    """Unit test for the `pulse.get_amplitude` function.

    This test verifies the correct calculation of the amplitude of step and square
    signals, both with and without specified regions of interest. It checks the
    amplitude for both positive and negative polarities.

    Test cases:
        - Step signal with positive polarity.
        - Step signal with negative polarity.
        - Square signal with positive polarity.
        - Square signal with negative polarity.

        - Step signal with custom initial and final values.
        - Square signal with custom initial and high values.
    """
    # positive polarity - step
    x, y_noisy = generate_step_signal(seed=0)
    amplitude = pulse.get_amplitude(x, y_noisy, (0, 2), (6, 8))
    check_scalar_result(
        "step, get step amplitude, positive polarity", amplitude, 5, atol=0.03
    )
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(signal, title=f"Step amplitude = {amplitude:.3f}")

    amplitude = pulse.get_amplitude(x, y_noisy)
    check_scalar_result(
        "step, get step amplitude, positive polarity", amplitude, 6.17, atol=0.03
    )
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(
        signal, title=f"Step amplitude (auto) = {amplitude:.3f}"
    )

    # negative polarity - step
    x, y_noisy = generate_step_signal(seed=0, y_initial=5, y_final=2)
    amplitude = pulse.get_amplitude(x, y_noisy, (0, 2), (6, 8))
    check_scalar_result(
        "step, get step amplitude, negative polarity", amplitude, 3, atol=0.03
    )
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(signal, title=f"Step amplitude (neg) = {amplitude:.3f}")

    # positive polarity - square
    x, y_noisy = generate_square_signal(seed=0)
    amplitude = pulse.get_amplitude(
        x, y_noisy, (0, 2), (12, 14), high_baseline_range=(5.5, 6.5)
    )
    check_scalar_result(
        "square, get square amplitude, positive polarity", amplitude, 5, atol=0.03
    )
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(signal, title=f"Square amplitude = {amplitude:.3f}")

    # negative polarity - square
    x, y_noisy = generate_square_signal(seed=0, y_initial=5, y_high=2)
    amplitude = pulse.get_amplitude(
        x, y_noisy, (0, 2), (12, 14), high_baseline_range=(5.5, 6.5)
    )
    check_scalar_result(
        "square, get square amplitude, negative polarity", amplitude, 3, atol=0.03
    )
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(
        signal, title=f"Square amplitude (neg) = {amplitude:.3f}"
    )

    # positive polarity - square
    x, y_noisy = generate_square_signal(seed=0)
    amplitude = pulse.get_amplitude(x, y_noisy, (0, 2), (12, 14))
    check_scalar_result(
        "square, get square amplitude, positive polarity", amplitude, 5, atol=0.6
    )
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(signal, title=f"Square amplitude = {amplitude:.3f}")

    # negative polarity - square
    x, y_noisy = generate_square_signal(seed=0, y_initial=5, y_high=2)
    amplitude = pulse.get_amplitude(x, y_noisy, (0, 2), (12, 14))
    check_scalar_result(
        "square, get square amplitude, negative polarity", amplitude, 3, atol=0.6
    )
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(
        signal, title=f"Square amplitude (neg) = {amplitude:.3f}"
    )


def test_get_crossing_ratio_time() -> None:
    """Unit test for the `pulse.get_crossing_ratio_time` function.

    This test verifies the correct calculation of the crossing time at a given ratio
    for both positive and negative polarity step signals, with and without specified
    baseline regions.
    """
    # positive polarity
    x, y_noisy = generate_step_signal(seed=0)

    crossing_time = pulse.get_crossing_ratio_time(x, y_noisy, (0, 2), (6, 8), 0.2)
    check_scalar_result(
        "step, get crossing time, positive polarity", crossing_time, 3.3761
    )
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(signal, title=f"Crossing time = {crossing_time:.3f}")

    crossing_time = pulse.get_crossing_ratio_time(x, y_noisy, None, None, 0.2)
    check_scalar_result(
        "step, get crossing time, positive polarity", crossing_time, 2.189, atol=0.001
    )
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(
        signal, title=f"Crossing time (auto) = {crossing_time:.3f}"
    )

    # negative polarity
    x, y_noisy = generate_step_signal(
        seed=0, y_initial=5, y_final=2, noise_amplitude=0.1
    )

    crossing_time = pulse.get_crossing_ratio_time(x, y_noisy, (0, 2), (6, 8), 0.2)
    check_scalar_result(
        "step, get crossing time, negative polarity", crossing_time, 3.31, atol=0.001
    )
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(
        signal, title=f"Crossing time (neg) = {crossing_time:.3f}"
    )

    x, y_noisy = generate_step_signal(
        seed=0, y_initial=5, y_final=2, noise_amplitude=0.2
    )


def test_get_step_rise_time() -> None:
    """Unit test for the `pulse.get_step_rise_time` function.

    This test verifies the correct calculation of the rise time for step signals with
    both positive and negative polarity, using various noise amplitudes and different
    ways of specifying rise ratios and baseline regions.
    """
    # positive polarity
    x, y_noisy = generate_step_signal(seed=0)

    rise_time = pulse.get_step_rise_time(x, y_noisy, (0, 2), (6, 8), 0.2, 0.2)
    check_scalar_result(
        "step, get rise time, positive polarity", rise_time, 1.19, atol=0.01
    )
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(signal, title=f"Rise time = {rise_time:.3f}")

    x, y_noisy = generate_step_signal(seed=0, noise_amplitude=0)
    rise_time = pulse.get_step_rise_time(x, y_noisy, (0, 2), (6, 8), 0.2, 0.2)
    check_scalar_result(
        "step, get rise time, positive polarity", rise_time, 1.19, atol=0.01
    )
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(signal, title=f"Rise time = {rise_time:.3f}")

    rise_time = pulse.get_step_rise_time(
        x, y_noisy, start_rise_ratio=0.2, stop_rise_ratio=0.2
    )
    check_scalar_result(
        "step, get rise time, positive polarity", rise_time, 1.19, atol=0.01
    )
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(signal, title=f"Rise time = {rise_time:.3f}")

    # negative polarity
    x, y_noisy = generate_step_signal(seed=0, y_initial=5, y_final=2)

    rise_time = pulse.get_step_rise_time(x, y_noisy, (0, 2), (6, 8), 0.2, 0.2)
    check_scalar_result(
        "step, get rise time, negative polarity", rise_time, 2.16, atol=0.01
    )
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(signal, title=f"Rise time (neg) = {rise_time:.3f}")


def test_get_step_time_at_half_maximum() -> None:
    """Unit test for the `pulse.get_crossing_ratio_time` function at 0.5 ratio.

    This test verifies the correct calculation of the time at which a step signal
    reaches half of its maximum amplitude, for both positive and negative polarity
    signals, using specified baseline regions.
    """
    # positive polarity
    x, y_noisy = generate_step_signal(seed=0)

    time = pulse.get_crossing_ratio_time(x, y_noisy, (0, 2), (6, 8), 0.5)
    check_scalar_result(
        "step, get time at half maximum, positive polarity", time, 3.95, atol=0.01
    )
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(signal, title=f"Time at half max = {time:.3f}")

    # negative polarity
    x, y_noisy = generate_step_signal(seed=0, y_initial=5, y_final=2)

    time = pulse.get_crossing_ratio_time(x, y_noisy, (0, 2), (6, 8), 0.5)
    check_scalar_result(
        "step, get time at half maximum, negative polarity", time, 3.80, atol=0.01
    )
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(signal, title=f"Time at half max (neg) = {time:.3f}")


def test_get_step_starttime() -> None:
    """Unit test for the `pulse.get_crossing_ratio_time` function with start ratio.

    This test verifies the correct calculation of the start time for step signals
    with both positive and negative polarity, using specified baseline regions and
    rise ratios.
    """
    # positive polarity
    x, y_noisy = generate_step_signal(seed=0)

    time = pulse.get_crossing_ratio_time(x, y_noisy, (0, 2), (6, 8), 0.2)
    check_scalar_result("step, get start time", time, 3.38, atol=0.01)
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(signal, title=f"Start time = {time:.3f}")

    # negative polarity
    x, y_noisy = generate_step_signal(seed=0, y_initial=5, y_final=2)

    time = pulse.get_crossing_ratio_time(x, y_noisy, (0, 2), (6, 8), 0.2)
    check_scalar_result("step, get start time", time, 2.38, atol=0.01)
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(signal, title=f"Start time (neg) = {time:.3f}")


def test_get_step_end_time() -> None:
    """Unit test for the `pulse.get_crossing_ratio_time` function with end ratio.

    This test verifies the correct calculation of the end time for step signals
    with both positive and negative polarity, using specified baseline regions and
    rise ratios. Using 1-0.2 = 0.8 for end crossing.
    """
    # positive polarity
    x, y_noisy = generate_step_signal(seed=0)
    time = pulse.get_crossing_ratio_time(x, y_noisy, (0, 2), (6, 8), 0.8)
    check_scalar_result("step, get end time", time, 4.57, atol=0.01)
    signal = create_signal("", x, y_noisy)
    guiutils.view_curves_if_gui(signal, title=f"End time = {time:.3f}")

    # negative polarity
    x, y_noisy = generate_step_signal(seed=0, y_initial=5, y_final=2)
    time = pulse.get_crossing_ratio_time(x, y_noisy, (0, 2), (6, 8), 0.8)
    check_scalar_result("step, get end time", time, 4.54, atol=0.01)
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
    # Use start_basement_range before the step, end_basement_range after
    start_basement_range = (0, 4)
    start_basement_level = np.mean(y[start_basement_range[0] : start_basement_range[1]])
    end_basement_range = (6, 8)
    start_rise_ratio = 0.1

    foot_info = pulse.get_foot_info(
        x,
        y,
        start_basement_range,
        end_basement_range,
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
        atol=start_basement_level,
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


def test_get_parameters() -> None:
    """Test the extraction of step parameters from generated step and square signals."""
    # Test the extraction of step parameters from a generated step signal
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
    # Use start_basement_range before the step, end_basement_range after
    start_basement_range = (0, 4)
    end_basement_range = (6, 8)
    start_rise_ratio = 0.1

    params = pulse.get_parameters(
        x,
        y,
        start_basement_range,
        end_basement_range,
        start_rise_ratio=start_rise_ratio,
    )

    # Check keys exist
    assert "polarity" in params
    assert "amplitude" in params
    assert "rise_time" in params
    assert "fall_time" in params
    assert "fwhm" in params
    assert "offset" in params
    assert "t50" in params
    assert "tmax" in params
    assert "foot_duration" in params

    # Check values for all keys
    expected_values = {
        "polarity": 1,
        "amplitude": 5,
        "rise_time": 0.79,
        "fall_time": None,
        "fwhm": None,
        "offset": 1,
        "t50": 4.5,
        "tmax": 5,
        "foot_duration": 4.105,
    }
    tolerances = {
        "polarity": 1e-8,
        "amplitude": 0.01,
        "rise_time": 0.01,
        "fall_time": 0.01,
        "fwhm": 0.01,
        "offset": 0.01,
        "t50": 0.01,
        "tmax": 0.01,
        "foot_duration": 0.01,
    }
    for key, expected in expected_values.items():
        if expected is not None:
            check_scalar_result(f"{key}", params[key], expected, atol=tolerances[key])
        else:
            assert params[key] is None, (
                f"Expected {key} to be None, but got {params[key]}"
            )
        assert params["signal_shape"] == SignalShape.STEP, (
            f"Expected signal_shape to be 'step', but got {params['signal_shape']}"
        )
        signal = create_signal("", x, y)
    guiutils.view_curves_if_gui(
        signal,
        title=(
            "Step parameters: "
            f"polarity={params['polarity']}, "
            f"offset={params['offset']}, "
            f"fwhm={params['fwhm']}, "
            f"rise_time={params['rise_time']}, "
            f"fall_time={params['fall_time']}, "
            f"t50={params['t50']}, "
            f"tmax={params['tmax']}, "
            f"foot_duration={params['foot_duration']}"
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
    # Use start_basement_range before the step, end_basement_range after
    start_basement_range = (0, 2.5)
    end_basement_range = (15, 17)
    start_rise_ratio = 0.1

    params = pulse.get_parameters(
        x,
        y,
        start_basement_range,
        end_basement_range,
        start_rise_ratio=start_rise_ratio,
    )

    # Check keys exist
    assert "polarity" in params
    assert "amplitude" in params
    assert "rise_time" in params
    assert "fall_time" in params
    assert "fwhm" in params
    assert "offset" in params
    assert "t50" in params
    assert "tmax" in params
    assert "foot_duration" in params

    # Check values for all keys
    expected_values = {
        "polarity": 1,
        "amplitude": 5,
        "rise_time": 0.79,
        "fall_time": 5.595,
        "fwhm": 6,
        "offset": 1,
        "t50": 4.5,
        "tmax": 5,
        "foot_duration": 4.105,
    }
    tolerances = {
        "polarity": 1e-8,
        "amplitude": 0.01,
        "rise_time": 0.01,
        "fall_time": 0.01,
        "fwhm": 0.01,
        "offset": 0.01,
        "t50": 0.01,
        "tmax": 0.01,
        "foot_duration": 0.01,
    }
    for key, expected in expected_values.items():
        if expected is not None:
            check_scalar_result(f"{key}", params[key], expected, atol=tolerances[key])
        else:
            assert params[key] is None, (
                f"Expected {key} to be None, but got {params[key]}"
            )
        assert params["signal_shape"] == SignalShape.SQUARE, (
            f"Expected signal_shape to be 'square', but got {params['signal_shape']}"
        )
        signal = create_signal("", x, y)
    guiutils.view_curves_if_gui(
        signal,
        title="Step parameters: "
        f"polarity={params['polarity']}, "
        f"offset={params['offset']}, "
        f"fwhm={params['fwhm']}, "
        f"rise_time={params['rise_time']}, "
        f"fall_time={params['fall_time']}, "
        f"t50={params['t50']}, "
        f"tmax={params['tmax']}, "
        f"foot_duration={params['foot_duration']}",
    )


@pytest.mark.validation
def test_signal_get_parameters() -> None:
    """Validation test for get_parameters computation function."""
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

    p = ParametersParam()
    p.start_basement_range_min = 0
    p.start_basement_range_max = 4
    p.end_basement_range_min = 6
    p.end_basement_range_max = 8
    p.start_rise_ratio = 0.1
    p.stop_rise_ratio = 0.1

    result = get_parameters(sig, p)

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

    signal = create_signal("", x, y)
    guiutils.view_curves_if_gui(
        signal,
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

    # Validation test for get_parameters with a SQUARE signal
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

    p = ParametersParam()
    p.start_basement_range_min = 0
    p.start_basement_range_max = 2.5
    p.end_basement_range_min = 15
    p.end_basement_range_max = 17
    p.start_rise_ratio = 0.1
    p.stop_rise_ratio = 0.1

    result = get_parameters(sig, p)

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
    signal = create_signal("", x, y)
    guiutils.view_curves_if_gui(
        signal,
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
    test_heuristically_recognize_shape()
    test_detect_polarity()
    test_get_amplitude()
    test_get_crossing_ratio_time()
    test_get_step_rise_time()
    test_get_step_time_at_half_maximum()
    test_get_step_starttime()
    test_get_step_end_time()
    test_heuristically_find_foot_end_time()
    test_get_foot_info()
    test_get_parameters()
    test_signal_get_parameters()
