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

import guidata.dataset as gds
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


class SumPixelsAlongAxisParam(gds.DataSet):
    """Parameters for summing an image along one axis.

    Attributes:
        axis: Axis to sum along, "X" sums across columns (result indexed by Y),
            "Y" sums across rows (result indexed by X).
    """

    AXES = (("X", "X"), ("Y", "Y"))
    axis = gds.ChoiceItem(_("Axis"), AXES, default="X")


@computation_function()
def sum_pixels_along_axis(image: ImageObj, p: SumPixelsAlongAxisParam) -> SignalObj:
    """Compute the sum of image values along the chosen axis and return a 1-D signal.

    The function sums the image along the X or Y axis (as selected in the
    parameters) and returns a signal object containing the summed values.
    The output X coordinates are built from the image spatial sampling (dx, dy)
    and are centered on the image physical center.

    Args:
        image: Input image object.
        p: Parameters selecting the axis to sum over.

    Returns:
        Signal object containing the summed profile.

    Raises:
        ValueError: if the axis parameter is invalid.
    """
    axis_map = {"X": 1, "Y": 0}
    try:
        axis = axis_map[p.axis]
    except KeyError as exc:
        raise ValueError(f"Invalid axis: {p.axis!s}") from exc

    # Use masked-aware sum to preserve mask semantics
    summed = ma.sum(image.data, axis=axis)

    # Convert masked result to plain ndarray (masked entries -> NaN)
    if ma.isMaskedArray(summed):
        if not (
            np.issubdtype(np.asarray(summed).dtype, np.floating)
            or np.issubdtype(np.asarray(summed).dtype, np.complexfloating)
        ):
            # Convert integer masked array to float to allow NaN
            summed = summed.astype(float)
        y = np.asarray(ma.filled(summed, np.nan))
    else:
        y = np.asarray(summed)
    # Image spatial info (fall back to sensible defaults)
    shape = image.data.shape
    x0 = image.x0
    y0 = image.y0
    dx = image.dx
    dy = image.dy

    # Build X coordinates: when summing along "X" (axis=1) the result is indexed by Y,
    # when summing along "Y" (axis=0) the result is indexed by X.
    if p.axis == "X":
        length = shape[0]
        # pixel centers, then shift so coordinates are centered on image physical center
        center = y0 + dy * length / 2.0
        x = y0 + (np.arange(length) + 0.5) * dy - center
        xunit = image.yunit
        xlabel = image.ylabel
    else:
        length = shape[1]
        center = x0 + dx * length / 2.0
        x = x0 + (np.arange(length) + 0.5) * dx - center
        xunit = image.xunit
        xlabel = image.xlabel

    # Data unit: prefer zunit (pixel value unit) then fallback to generic unit/meta
    zunit = image.zunit
    ylabel = image.zlabel

    # Create the signal providing both x and y and pass units/labels
    dst_signal = create_signal(
        title=f"{image.title} - Sum along {p.axis}",
        x=x.astype(float),
        y=y,
        units=(xunit, zunit),
        labels=(xlabel, ylabel),
    )

    # Keep source metadata if present
    try:
        source = image.get_metadata_option("source")
        dst_signal.set_metadata_option("source", source)
    except ValueError:
        pass

    return dst_signal
