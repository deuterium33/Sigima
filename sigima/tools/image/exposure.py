# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Exposure and level adjustment module
------------------------------------

This module provides functions for adjusting image exposure, contrast, and intensity
levels.

Features include:

- Dynamic range scaling and adjustment
- Various normalization methods (maximum, amplitude, area, energy, RMS)
- Data type preserving transformations

These tools support image enhancement and preprocessing operations that adjust
the intensity distribution of images while preserving their essential characteristics.
"""

from __future__ import annotations

import numpy as np

from sigima.enums import NormalizationMethod
from sigima.tools.checks import check_2d_array
from sigima.tools.image.preprocessing import scale_data_to_min_max


@check_2d_array(non_constant=True)
def normalize(
    data: np.ndarray,
    parameter: NormalizationMethod = NormalizationMethod.MAXIMUM,
) -> np.ndarray:
    """Normalize input array to a given parameter.

    Args:
        data: Input data
        parameter: Normalization parameter (default: MAXIMUM)

    Returns:
        Normalized array
    """
    if parameter == NormalizationMethod.MAXIMUM:
        return scale_data_to_min_max(data, np.nanmin(data) / np.nanmax(data), 1.0)
    if parameter == NormalizationMethod.AMPLITUDE:
        return scale_data_to_min_max(data, 0.0, 1.0)
    fdata = np.array(data, dtype=float)
    if parameter == NormalizationMethod.AREA:
        return fdata / np.nansum(fdata)
    if parameter == NormalizationMethod.ENERGY:
        return fdata / np.sqrt(np.nansum(fdata * fdata.conjugate()))
    if parameter == NormalizationMethod.RMS:
        return fdata / np.sqrt(np.nanmean(fdata * fdata.conjugate()))
    raise ValueError(f"Unsupported parameter {parameter}")


@check_2d_array
def flatfield(
    rawdata: np.ndarray, flatdata: np.ndarray, threshold: float | None = None
) -> np.ndarray:
    """Compute flat-field correction

    Args:
        rawdata: Raw data
        flatdata: Flat-field data
        threshold: Threshold for flat-field correction (default: None)

    Returns:
        Flat-field corrected data
    """
    dtemp = np.array(rawdata, dtype=float, copy=True) * np.nanmean(flatdata)
    dunif = np.array(flatdata, dtype=float, copy=True)
    dunif[dunif == 0] = 1.0
    dcorr_all = np.array(dtemp / dunif, dtype=rawdata.dtype)
    dcorr = np.array(rawdata, copy=True)
    dcorr[rawdata > threshold] = dcorr_all[rawdata > threshold]
    return dcorr
