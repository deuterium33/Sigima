"""
Signal/Image Data Extraction
-----------------------------

This module contains functions for extracting information and features from image data:

- Radial profile extraction
- Statistical analysis and feature extraction functions
- Data analysis utilities
"""

from __future__ import annotations

import numpy as np

from sigima.tools.checks import check_2d_array


@check_2d_array
def get_radial_profile(
    data: np.ndarray, center: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return radial profile of image data

    Args:
        data: Input data (2D array)
        center: Coordinates of the center of the profile (x, y)

    Returns:
        Radial profile (X, Y) where X is the distance from the center (1D array)
        and Y is the average value of pixels at this distance (1D array)
    """
    y, x = np.indices((data.shape))  # Get the indices of pixels
    x0, y0 = center
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)  # Calculate distance to the center
    r = r.astype(int)

    # Average over the same distance
    tbin = np.bincount(r.ravel(), data.ravel())  # Sum of pixel values at each distance
    nr = np.bincount(r.ravel())  # Number of pixels at each distance

    yprofile = tbin / nr  # this is the half radial profile
    # Let's mirror it to get the full radial profile (the first element is the center)
    yprofile = np.concatenate((yprofile[::-1], yprofile[1:]))
    # The x axis is the distance from the center (0 is the center)
    xprofile = np.arange(len(yprofile)) - len(yprofile) // 2

    return xprofile, yprofile
