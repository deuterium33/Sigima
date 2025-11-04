# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Windowing (see parent package :mod:`sigima.algorithms.signal`)

"""

from __future__ import annotations

import numpy as np
import scipy.signal.windows

from sigima.enums import WindowingMethod

WINDOWING_FUNCTIONS_MAP = {
    WindowingMethod.BARTHANN: scipy.signal.windows.barthann,
    WindowingMethod.BARTLETT: np.bartlett,
    WindowingMethod.BLACKMAN: np.blackman,
    WindowingMethod.BLACKMAN_HARRIS: scipy.signal.windows.blackmanharris,
    WindowingMethod.BOHMAN: scipy.signal.windows.bohman,
    WindowingMethod.BOXCAR: scipy.signal.windows.boxcar,
    WindowingMethod.COSINE: scipy.signal.windows.cosine,
    WindowingMethod.EXPONENTIAL: scipy.signal.windows.exponential,
    WindowingMethod.FLAT_TOP: scipy.signal.windows.flattop,
    WindowingMethod.GAUSSIAN: scipy.signal.windows.gaussian,
    WindowingMethod.HAMMING: np.hamming,
    WindowingMethod.HANN: np.hanning,
    WindowingMethod.KAISER: np.kaiser,
    WindowingMethod.LANCZOS: scipy.signal.windows.lanczos,
    WindowingMethod.NUTTALL: scipy.signal.windows.nuttall,
    WindowingMethod.PARZEN: scipy.signal.windows.parzen,
    WindowingMethod.TAYLOR: scipy.signal.windows.taylor,
    WindowingMethod.TUKEY: scipy.signal.windows.tukey,
}

assert set(WINDOWING_FUNCTIONS_MAP.keys()) == set(WindowingMethod), (
    f"WINDOWING_FUNCTIONS_MAP must contain all WindowingMethod enum values. "
    f"Missing: {set(WindowingMethod) - set(WINDOWING_FUNCTIONS_MAP.keys())}, "
    f"Extra: {set(WINDOWING_FUNCTIONS_MAP.keys()) - set(WindowingMethod)}"
)


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
    win_func = WINDOWING_FUNCTIONS_MAP[method]
    # Cases with parameters:
    if method == WindowingMethod.GAUSSIAN:
        return y * win_func(len(y), sigma)
    if method == WindowingMethod.KAISER:
        return y * win_func(len(y), beta)
    if method == WindowingMethod.TUKEY:
        return y * win_func(len(y), alpha)
    # Cases without parameters:
    return y * win_func(len(y))
