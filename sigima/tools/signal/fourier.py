# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Fourier Analysis (see parent package :mod:`sigima.tools.signal`).

"""

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.signal  # type: ignore[import]

from sigima.tools.checks import check_1d_arrays
from sigima.tools.signal.dynamic import sampling_rate


@check_1d_arrays(x_evenly_spaced=True)
def zero_padding(
    x: np.ndarray, y: np.ndarray, n_prepend: int = 0, n_append: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Prepend and append zeros.

    This function pads the input signal with zeros at the beginning and end.

    Args:
        x: X data.
        y: Y data.
        n_prepend: Number of zeros to prepend.
        n_append: Number of zeros to append.

    Returns:
        Tuple (xnew, ynew): Padded x and y.
    """
    if n_prepend < 0:
        raise ValueError("Number of zeros to prepend must be non-negative.")
    if n_append < 0:
        raise ValueError("Number of zeros to append must be non-negative.")

    dx = np.mean(np.diff(x))
    xnew = np.linspace(
        x[0] - n_prepend * dx,
        x[-1] + n_append * dx,
        y.size + n_prepend + n_append,
    )
    ynew = np.pad(y, (n_prepend, n_append), mode="constant")
    return xnew, ynew


@check_1d_arrays(x_evenly_spaced=True)
def fft1d(
    x: np.ndarray, y: np.ndarray, shift: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Fast Fourier Transform (FFT) of a 1D real signal.

    Args:
        x: Time domain axis (evenly spaced).
        y: Signal values.
        shift: If True, shift zero frequency and its corresponding FFT component to the
        center.

    Returns:
        Tuple (f, sp): Frequency axis and corresponding FFT values.
    """
    dt = np.mean(np.diff(x))
    f = np.fft.fftfreq(x.size, d=dt)  # Frequency axis
    sp = np.fft.fft(y)  # Spectrum values
    if shift:
        f = np.fft.fftshift(f)
        sp = np.fft.fftshift(sp)
    return f, sp


@check_1d_arrays(x_evenly_spaced=False, x_sorted=False, y_dtype=np.complexfloating)
def ifft1d(
    f: np.ndarray, sp: np.ndarray, initial: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the inverse Fast Fourier Transform (FFT) of a 1D complex spectrum.

    Args:
        f: Frequency axis (evenly spaced).
        sp: FFT values.
        initial: Starting value for the time axis.

    Returns:
        Tuple (x, y): Time axis and real signal.

    Raises:
        ValueError: If frequency array is not evenly spaced or has fewer than 2 points.
    """
    if f.size < 2:
        raise ValueError("Frequency array must have at least two elements.")

    if np.all(np.diff(f) >= 0.0):
        # If frequencies are sorted, assume input is shifted.
        # The spectrum needs to be unshifted.
        sp = np.fft.ifftshift(sp)
    else:
        # Otherwise assume input is not shifted.
        # The frequencies need to be shifted.
        f = np.fft.fftshift(f)

    diff_f = np.diff(f)
    df = np.mean(diff_f)
    if not np.allclose(diff_f, df):
        raise ValueError("Frequency array must be evenly spaced.")

    y = np.fft.ifft(sp)
    dt = 1.0 / (f.size * df)
    x = np.linspace(initial, initial + (y.size - 1) * dt, y.size)

    return x, y.real


@check_1d_arrays(x_evenly_spaced=True)
def magnitude_spectrum(
    x: np.ndarray, y: np.ndarray, decibel: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Compute magnitude spectrum.

    Args:
        x: X data.
        y: Y data.
        decibel: Compute the magnitude spectrum root-power level in decibel (dB).

    Returns:
        Tuple (f, mag_spectrum): Frequency values and magnitude spectrum.
    """
    f, spectrum = fft1d(x, y)
    mag_spectrum = np.abs(spectrum)
    if decibel:
        mag_spectrum = 20 * np.log10(mag_spectrum)
    return f, mag_spectrum


@check_1d_arrays(x_evenly_spaced=True)
def phase_spectrum(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute phase spectrum.

    Args:
        x: X data.
        y: Y data.

    Returns:
        Tuple (f, phase): Frequency values and phase spectrum in degrees.
    """
    f, spectrum = fft1d(x, y)
    phase = np.rad2deg(np.angle(spectrum))
    return f, phase


@check_1d_arrays(x_evenly_spaced=True)
def psd(
    x: np.ndarray, y: np.ndarray, decibel: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the Power Spectral Density (PSD) using Welch's method.

    Args:
        x: X data.
        y: Y data.
        decibel: Compute the power spectral density power level in decibel (dB).

    Returns:
        Tuple (f, welch_psd): Frequency values and PSD.
    """
    f, welch_psd = scipy.signal.welch(y, fs=sampling_rate(x))
    if decibel:
        welch_psd = 10 * np.log10(welch_psd)
    return f, welch_psd


@check_1d_arrays(x_evenly_spaced=True)
def brickwall_filter(
    x: np.ndarray,
    y: np.ndarray,
    mode: Literal["lowpass", "highpass", "bandpass", "bandstop"],
    cutoff0: float,
    cutoff1: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply an ideal frequency filter ("brick wall" filter) to a signal.

    Args:
        x: Time domain axis (evenly spaced).
        y: Signal values (same length as x).
        mode: Type of filter to apply.
        cutoff0: First cutoff frequency.
        cutoff1: Second cutoff frequency, required for band-pass and band-stop filters.

    Returns:
        Tuple (x, y_filtered), where y_filtered is the filtered signal.

    Raises:
        ValueError: If mode is unknown.
        ValueError: If cutoff0 is not positive.
        ValueError: If cutoff1 is missing for band-pass and band-stop filters.
        ValueError: If cutoff0 > cutoff1 for band-pass and band-stop filters.
    """
    if mode not in ("lowpass", "highpass", "bandpass", "bandstop"):
        raise ValueError(f"Unknown filter mode: {mode!r}")

    if cutoff0 <= 0.0:
        raise ValueError("Cutoff frequency must be positive.")

    if mode in ("bandpass", "bandstop"):
        if cutoff1 is None:
            raise ValueError(f"cut1 must be specified for mode '{mode}'")
        if cutoff0 > cutoff1:
            raise ValueError("cut0 must be less than or equal to cut1.")

    freqs, ffty = fft1d(x, y, shift=False)

    if mode == "lowpass":
        frequency_mask = np.abs(freqs) <= cutoff0
    elif mode == "highpass":
        frequency_mask = np.abs(freqs) >= cutoff0
    elif mode == "bandpass":
        frequency_mask = (np.abs(freqs) >= cutoff0) & (np.abs(freqs) <= cutoff1)
    else:  # bandstop
        frequency_mask = (np.abs(freqs) <= cutoff0) | (np.abs(freqs) >= cutoff1)

    ffty_filtered = ffty * frequency_mask
    _, y_filtered = ifft1d(freqs, ffty_filtered)
    y_filtered = y_filtered.real
    return x, y_filtered
