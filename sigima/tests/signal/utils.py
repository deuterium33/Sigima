# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.
"""
.. Utility functions for generating synthetic step and square signals with optional
noise.

"""

from __future__ import annotations

import numpy as np


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
    """
    Generate a noisy step signal with a linear rise.

    The function creates a time vector and generates a signal that starts at
    `y_initial`, rises linearly to `y_final` starting at `t_step` over a duration
    of `t_rise`, and remains at `y_final` afterwards.
    Gaussian noise is added to the signal.

    Parameters
    ----------
    t_start:
        Start time of the signal (default is 0).
    t_end:
        End time of the signal (default is 10).
    dt:
        Time step for the time vector (default is 0.01).
    t_rise:
        Duration of the linear rise from `y_initial` to `y_final` (default is 2).
    t_step:
        Time at which the step (rise) begins (default is 3).
    y_initial:
        Initial value of the signal before the step (default is 0).
    y_final:
        Final value of the signal after the rise (default is 5).
    noise_amplitude:
        Standard deviation of the Gaussian noise added to the signal (default is 0.2).
    seed:
        Seed for the random number generator for reproducibility (default is None).

    Returns
    -------
    x: Time vector.
    y_noisy: Noisy step signal.
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
    """
    Generates a synthetic square-like signal with configurable rise, plateau, and fall
    times, and adds Gaussian noise.
    Parameters
    ----------
    t_start : Start time of the signal.
    t_end : End time of the signal.
    dt : Time step for the signal.
    t_rise : Duration of the rising edge.
    t_step : Time at which the rising edge starts.
    square_duration : Duration of the high (plateau) part of the signal.
    t_fall : Duration of the falling edge.
    y_initial : Initial (and final) value of the signal.
    y_high : Value of the signal during the plateau.
    noise_amplitude : Standard deviation of the Gaussian noise added to the signal.
    seed : Seed for the random number generator.

    Returns
    -------
    x: Time vector.
    y_noisy: Noisy step signal.
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
