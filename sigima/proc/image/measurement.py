# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Measurement computation module
------------------------------

This module provides tools for extracting quantitative information from images,
such as object centroids, enclosing circles, and region-based statistics.

Main features include:

- Centroid and enclosing circle computation
- Region/property measurements
- Statistical analysis of image regions

These functions are useful for image quantification and morphometric analysis.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

# Note:
# ----
# - All `guidata.dataset.DataSet` parameter classes must also be imported
#   in the `sigima.params` module.
# - All functions decorated by `computation_function` must be imported in the upper
#   level `sigima.proc.image` module.

from __future__ import annotations

import numpy as np
from numpy import ma

import sigima.tools.image
from sigima.config import _
from sigima.objects import (
    GeometryResult,
    ImageObj,
    SignalObj,
    TableResult,
    TableResultBuilder,
    create_signal,
)
from sigima.proc.decorator import computation_function
from sigima.proc.image.base import compute_geometry_from_obj

# NOTE: Only parameter classes DEFINED in this module should be included in __all__.
# Parameter classes imported from other modules (like sigima.proc.base) should NOT
# be re-exported to avoid Sphinx cross-reference conflicts. The sigima.params module
# serves as the central API point that imports and re-exports all parameter classes.
__all__ = [
    "centroid",
    "enclosing_circle",
    "stats",
]


def get_centroid_coords(data: np.ndarray) -> np.ndarray:
    """Return centroid coordinates
    with :py:func:`sigima.tools.image.get_centroid_auto`

    Args:
        data: input data

    Returns:
        Centroid coordinates
    """
    y, x = sigima.tools.image.get_centroid_auto(data)
    return np.array([(x, y)])


@computation_function()
def centroid(image: ImageObj) -> GeometryResult | None:
    """Compute centroid
    with :py:func:`sigima.tools.image.get_centroid_fourier`

    Args:
        image: input image

    Returns:
        Centroid coordinates
    """
    return compute_geometry_from_obj("centroid", "marker", image, get_centroid_coords)


def get_enclosing_circle_coords(data: np.ndarray) -> np.ndarray:
    """Return diameter coords for the circle contour enclosing image
    values above threshold (FWHM)

    Args:
        data: input data

    Returns:
        Diameter coords
    """
    x, y, r = sigima.tools.image.get_enclosing_circle(data)
    return np.array([[x, y, r]])


@computation_function()
def enclosing_circle(image: ImageObj) -> GeometryResult | None:
    """Compute minimum enclosing circle
    with :py:func:`sigima.tools.image.get_enclosing_circle`

    Args:
        image: input image

    Returns:
        Diameter coords
    """
    return compute_geometry_from_obj(
        "enclosing_circle", "circle", image, get_enclosing_circle_coords
    )


def __calc_snr_without_warning(data: np.ndarray) -> float:
    """Calculate SNR based on <z>/σ(z), ignoring warnings

    Args:
        data: input data

    Returns:
        Signal-to-noise ratio
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = ma.mean(data) / ma.std(data)
    return snr


@computation_function()
def stats(obj: ImageObj) -> TableResult:
    """Compute statistics on an image

    Args:
        obj: input image object

    Returns:
        Result properties
    """
    table = TableResultBuilder(_("Image statistics"))
    table.add(ma.min, "min", "min(z) = %g {.zunit}")
    table.add(ma.max, "max", "max(z) = %g {.zunit}")
    table.add(ma.mean, "mean", "<z> = %g {.zunit}")
    table.add(ma.median, "median", "median(z) = %g {.zunit}")
    table.add(ma.std, "std", "σ(z) = %g {.zunit}")
    table.add(__calc_snr_without_warning, "snr", "<z>/σ(z) = %g")
    table.add(ma.ptp, "ptp", "peak-to-peak(z) = %g {.zunit}")
    table.add(ma.sum, "sum", "Σ(z) = %g {.zunit}")
    return table.compute(obj)


@computation_function()
def sum_columns(image: ImageObj) -> SignalObj:
    """Compute the horizontal projection profile by summing values along the y-axis.

    Args:
        image: Input image object.

    Returns:
        Signal object containing the profile.
    """
    data = image.data
    assert data is not None
    length = data.shape[1]
    x0 = image.x0
    dx = image.dx
    assert x0 is not None
    assert dx is not None
    x = np.linspace(x0 + 0.5 * dx, x0 + (length - 0.5) * dx, length)
    xlabel = image.xlabel
    xunit = image.xunit
    assert xlabel is not None
    assert xunit is not None
    zlabel = image.zlabel
    zunit = image.zunit
    assert zlabel is not None
    assert zunit is not None
    source = image.metadata.get("source")
    metadata = {"source": source} if source is not None else {}
    title = "Horizontal projection"
    # Cast to np.float64 only if original image is of integer type
    dtype = np.float64 if np.issubdtype(data.dtype, np.integer) else data.dtype
    dst_signal = create_signal(
        title=title,
        x=x,
        y=data.sum(axis=0, dtype=dtype),
        metadata=metadata,
        units=(xunit, zunit),
        labels=(xlabel, zlabel),
    )
    return dst_signal


@computation_function()
def sum_rows(image: ImageObj) -> SignalObj:
    """Compute the vertical projection profile by summing values along the x-axis.

    Args:
        image: Input image object.

    Returns:
        Signal object containing the profile.
    """
    data = image.data
    assert data is not None
    length = data.shape[0]
    y0 = image.y0
    dy = image.dy
    assert y0 is not None
    assert dy is not None
    x = np.linspace(y0 + 0.5 * dy, y0 + (length - 0.5) * dy, length)
    xlabel = image.ylabel
    xunit = image.yunit
    assert xlabel is not None
    assert xunit is not None
    zlabel = image.zlabel
    zunit = image.zunit
    assert zlabel is not None
    assert zunit is not None
    source = image.metadata.get("source")
    metadata = {"source": source} if source is not None else {}
    title = "Vertical projection"
    # Cast to np.float64 only if original image is of integer type
    dtype = np.float64 if np.issubdtype(data.dtype, np.integer) else data.dtype
    dst_signal = create_signal(
        title=title,
        x=x,
        y=data.sum(axis=1, dtype=dtype),
        metadata=metadata,
        units=(xunit, zunit),
        labels=(xlabel, zlabel),
    )
    return dst_signal
