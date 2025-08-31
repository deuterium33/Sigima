# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Windowing (see parent package :mod:`sigima.algorithms.signal`)

"""

from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.signal.windows

from sigima.proc.enums import WindowingMethod


def get_window(method: WindowingMethod) -> Callable[[int], np.ndarray]:
    """Get the window function.

    .. note::

        The window functions are from `scipy.signal.windows` and `numpy`.
        All functions take an integer argument that specifies the length of the window,
        and return a NumPy array of the same length.

    Args:
        method: Windowing function method.

    Returns:
        Window function.

    Raises:
        ValueError: If the method is not recognized.
    """
    win_func = {
        WindowingMethod.BARTHANN: scipy.signal.windows.barthann,
        WindowingMethod.BARTLETT: np.bartlett,
        WindowingMethod.BLACKMAN: np.blackman,
        WindowingMethod.BLACKMAN_HARRIS: scipy.signal.windows.blackmanharris,
        WindowingMethod.BOHMAN: scipy.signal.windows.bohman,
        WindowingMethod.BOXCAR: scipy.signal.windows.boxcar,
        WindowingMethod.COSINE: scipy.signal.windows.cosine,
        WindowingMethod.EXPONENTIAL: scipy.signal.windows.exponential,
        WindowingMethod.FLAT_TOP: scipy.signal.windows.flattop,
        WindowingMethod.HAMMING: np.hamming,
        WindowingMethod.HANN: np.hanning,
        WindowingMethod.LANCZOS: scipy.signal.windows.lanczos,
        WindowingMethod.NUTTALL: scipy.signal.windows.nuttall,
        WindowingMethod.PARZEN: scipy.signal.windows.parzen,
        WindowingMethod.TAYLOR: scipy.signal.windows.taylor,
    }.get(method)
    if win_func is not None:
        return win_func
    raise ValueError(f"Invalid window type {method.value}")


def apply_window(
    y: np.ndarray,
    method: WindowingMethod = WindowingMethod.HAMMING,
    alpha: float = 0.5,
    beta: float = 14.0,
    sigma: float = 7.0,
) -> np.ndarray:
    """Apply windowing to the input data.

    Args:
        x: X data.
        y: Y data.
        method: Windowing function. Defaults to "HAMMING".
        alpha: Tukey window parameter. Defaults to 0.5.
        beta: Kaiser window parameter. Defaults to 14.0.
        sigma: Gaussian window parameter. Defaults to 7.0.

    Returns:
        Windowed Y data.

    Raises:
        ValueError: If the method is not recognized.
    """
    # Cases with parameters:
    if method == WindowingMethod.GAUSSIAN:
        return y * scipy.signal.windows.gaussian(len(y), sigma)
    if method == WindowingMethod.KAISER:
        return y * np.kaiser(len(y), beta)
    if method == WindowingMethod.TUKEY:
        return y * scipy.signal.windows.tukey(len(y), alpha)
    # Cases without parameters:
    win_func = get_window(method)
    return y * win_func(len(y))
