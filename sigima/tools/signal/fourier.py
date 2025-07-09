# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Fourier Analysis (see parent package :mod:`sigima.tools.signal`).

"""

from __future__ import annotations

import numpy as np
import scipy.signal  # type: ignore[import]

from sigima.tools.signal.decorator import SignalCheck
from sigima.tools.signal.dynamic import sampling_rate


@SignalCheck(x_evenly_spaced=True)
def zero_padding(
    x: np.ndarray, y: np.ndarray, n_prepend: int = 0, n_append: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Append n zeros at the end of the signal.

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


@SignalCheck(x_dtype=np.inexact, x_evenly_spaced=True)
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


@SignalCheck(
    x_dtype=np.inexact,
    x_sorted=False,
    x_evenly_spaced=False,
    y_dtype=np.complexfloating,
)
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


@SignalCheck(x_evenly_spaced=True)
def magnitude_spectrum(
    x: np.ndarray, y: np.ndarray, log_scale: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Compute magnitude spectrum.

    Args:
        x: X data
        y: Y data
        log_scale: Use log scale. Defaults to False.

    Returns:
        Magnitude spectrum (X data, Y data)
    """
    x1, y1 = fft1d(x, y)
    if log_scale:
        y_mag = 20 * np.log10(np.abs(y1))
    else:
        y_mag = np.abs(y1)
    return x1, y_mag


@SignalCheck(x_evenly_spaced=True)
def phase_spectrum(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute phase spectrum.

    Args:
        x: X data
        y: Y data

    Returns:
        Phase spectrum in degrees (X data, Y data)
    """
    x1, y1 = fft1d(x, y)
    y_phase = np.rad2deg(np.angle(y1))
    return x1, y_phase


@SignalCheck()
def psd(
    x: np.ndarray, y: np.ndarray, log_scale: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Power Spectral Density (PSD), using the Welch method.

    Args:
        x: X data
        y: Y data
        log_scale: Use log scale. Defaults to False.

    Returns:
        Power Spectral Density (PSD): X data, Y data (tuple)
    """
    x1, y1 = scipy.signal.welch(y, fs=sampling_rate(x))
    if log_scale:
        y1 = 10 * np.log10(y1)
    return x1, y1


@SignalCheck(x_evenly_spaced=True)
def sort_frequencies(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sort from X,Y data by computing FFT(y).

    Args:
        x: X data
        y: Y data

    Returns:
        Sorted frequencies in ascending order
    """
    freqs, fourier = fft1d(x, y, shift=False)
    return freqs[np.argsort(fourier)]
