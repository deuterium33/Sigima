# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Scaling (see parent package :mod:`sigima.algorithms.signal`)

"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import numpy as np

from sigima.enums import NormalizationMethod


def normalize(
    yin: np.ndarray,
    parameter: NormalizationMethod = NormalizationMethod.MAXIMUM,
) -> np.ndarray:
    """Normalize input array to a given parameter.

    Args:
        yin: Input array
        parameter: Normalization parameter. Defaults to MAXIMUM

    Returns:
        Normalized array
    """
    axis = len(yin.shape) - 1
    if parameter == NormalizationMethod.MAXIMUM:
        maximum = np.nanmax(yin, axis)
        if axis == 1:
            maximum = maximum.reshape((len(maximum), 1))
        maxarray = np.tile(maximum, yin.shape[axis]).reshape(yin.shape)
        return yin / maxarray
    if parameter == NormalizationMethod.AMPLITUDE:
        ytemp = np.array(yin, copy=True)
        minimum = np.nanmin(yin, axis)
        if axis == 1:
            minimum = minimum.reshape((len(minimum), 1))
        ytemp -= minimum
        return normalize(ytemp, parameter=NormalizationMethod.MAXIMUM)
    if parameter == NormalizationMethod.AREA:
        return yin / np.nansum(yin)
    if parameter == NormalizationMethod.ENERGY:
        return yin / np.sqrt(np.nansum(yin * yin.conjugate()))
    # At this point, we must have RMS normalization (last option)
    assert parameter == NormalizationMethod.RMS
    return yin / np.sqrt(np.nanmean(yin * yin.conjugate()))
