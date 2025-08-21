# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Geometry computation module
---------------------------

This module implements geometric transformations and manipulations for images,
such as rotations, flips, resizing, axis swapping, binning, and padding.

Main features include:
- Rotation by arbitrary or fixed angles
- Horizontal and vertical flipping
- Resizing and binning of images
- Axis swapping and zero padding

These functions are useful for preparing and augmenting image data.
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
import scipy.ndimage as spi

import sigima.tools.image
from sigima.config import _
from sigima.objects.image import ImageObj
from sigima.proc.base import dst_1_to_1
from sigima.proc.decorator import computation_function
from sigima.proc.transformations import transformer

__all__ = [
    "TranslateParam",
    "translate",
    "RotateParam",
    "rotate",
    "rotate90",
    "rotate270",
    "fliph",
    "flipv",
    "ResizeParam",
    "resize",
    "BinningParam",
    "binning",
    "transpose",
]


class TranslateParam(gds.DataSet):
    """Translate parameters"""

    dx = gds.FloatItem(_("X translation"), default=0.0)
    dy = gds.FloatItem(_("Y translation"), default=0.0)


@computation_function()
def translate(src: ImageObj, p: TranslateParam) -> ImageObj:
    """Translate data with :py:func:`scipy.ndimage.shift`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "translate", f"dx={p.dx}, dy={p.dy}")
    dst.x0 += p.dx
    dst.y0 += p.dy
    transformer.transform_roi(dst, "translate", dx=p.dx, dy=p.dy)
    return dst


class RotateParam(gds.DataSet):
    """Rotate parameters"""

    boundaries = ("constant", "nearest", "reflect", "wrap")
    prop = gds.ValueProp(False)

    angle = gds.FloatItem(f"{_('Angle')} (°)", default=0.0)
    mode = gds.ChoiceItem(
        _("Mode"), list(zip(boundaries, boundaries)), default=boundaries[0]
    )
    cval = gds.FloatItem(
        _("cval"),
        default=0.0,
        help=_(
            "Value used for points outside the "
            "boundaries of the input if mode is "
            "'constant'"
        ),
    )
    reshape = gds.BoolItem(
        _("Reshape the output array"),
        default=False,
        help=_(
            "Reshape the output array "
            "so that the input array is "
            "contained completely in the output"
        ),
    )
    prefilter = gds.BoolItem(_("Prefilter the input image"), default=True).set_prop(
        "display", store=prop
    )
    order = gds.IntItem(
        _("Order"),
        default=3,
        min=0,
        max=5,
        help=_("Spline interpolation order"),
    ).set_prop("display", active=prop)


@computation_function()
def rotate(src: ImageObj, p: RotateParam) -> ImageObj:
    """Rotate data with :py:func:`scipy.ndimage.rotate`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "rotate", f"α={p.angle:.3f}°, mode='{p.mode}'")
    dst.data = spi.rotate(
        src.data,
        p.angle,
        reshape=p.reshape,
        order=p.order,
        mode=p.mode,
        cval=p.cval,
        prefilter=p.prefilter,
    )
    dst.roi = None  # Reset ROI as it may change after rotation
    return dst


@computation_function()
def rotate90(src: ImageObj) -> ImageObj:
    """Rotate data 90° with :py:func:`numpy.rot90`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "rotate90")
    dst.data = np.rot90(src.data)
    transformer.transform_roi(dst, "rotate", angle=-np.pi / 2, center=(dst.xc, dst.yc))
    return dst


@computation_function()
def rotate270(src: ImageObj) -> ImageObj:
    """Rotate data 270° with :py:func:`numpy.rot90`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "rotate270")
    dst.data = np.rot90(src.data, 3)
    transformer.transform_roi(dst, "rotate", angle=np.pi / 2, center=(dst.xc, dst.yc))
    return dst


@computation_function()
def fliph(src: ImageObj) -> ImageObj:
    """Flip data horizontally with :py:func:`numpy.fliplr`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "fliph")
    dst.data = np.fliplr(src.data)
    transformer.transform_roi(dst, "fliph", cx=dst.xc)
    return dst


@computation_function()
def flipv(src: ImageObj) -> ImageObj:
    """Flip data vertically with :py:func:`numpy.flipud`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "flipv")
    dst.data = np.flipud(src.data)
    transformer.transform_roi(dst, "flipv", cy=dst.yc)
    return dst


class ResizeParam(gds.DataSet):
    """Resize parameters"""

    boundaries = ("constant", "nearest", "reflect", "wrap")
    prop = gds.ValueProp(False)

    zoom = gds.FloatItem(_("Zoom"), default=1.0)
    mode = gds.ChoiceItem(
        _("Mode"), list(zip(boundaries, boundaries)), default=boundaries[0]
    )
    cval = gds.FloatItem(
        _("cval"),
        default=0.0,
        help=_(
            "Value used for points outside the "
            "boundaries of the input if mode is "
            "'constant'"
        ),
    )
    prefilter = gds.BoolItem(_("Prefilter the input image"), default=True).set_prop(
        "display", store=prop
    )
    order = gds.IntItem(
        _("Order"),
        default=3,
        min=0,
        max=5,
        help=_("Spline interpolation order"),
    ).set_prop("display", active=prop)


@computation_function()
def resize(src: ImageObj, p: ResizeParam) -> ImageObj:
    """Zooming function with :py:func:`scipy.ndimage.zoom`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "resize", f"zoom={p.zoom:.3f}")
    dst.data = spi.zoom(
        src.data,
        p.zoom,
        order=p.order,
        mode=p.mode,
        cval=p.cval,
        prefilter=p.prefilter,
    )
    if dst.dx is not None and dst.dy is not None:
        dst.dx, dst.dy = dst.dx / p.zoom, dst.dy / p.zoom
    return dst


class BinningParam(gds.DataSet):
    """Binning parameters."""

    sx = gds.IntItem(
        _("Cluster size (X)"),
        default=2,
        min=2,
        help=_("Number of adjacent pixels to be combined together along X-axis."),
    )
    sy = gds.IntItem(
        _("Cluster size (Y)"),
        default=2,
        min=2,
        help=_("Number of adjacent pixels to be combined together along Y-axis."),
    )
    operations = sigima.tools.image.BINNING_OPERATIONS
    operation = gds.ChoiceItem(
        _("Operation"),
        list(zip(operations, operations)),
        default=operations[0],
    )
    dtypes = ["dtype"] + ImageObj.get_valid_dtypenames()
    dtype_str = gds.ChoiceItem(
        _("Data type"),
        list(zip(dtypes, dtypes)),
        help=_("Output image data type."),
    )
    change_pixel_size = gds.BoolItem(
        _("Change pixel size"),
        default=True,
        help=_(
            "If checked, pixel size is updated according to binning factors. "
            "Users who prefer to work with pixel coordinates may want to uncheck this."
        ),
    )


@computation_function()
def binning(src: ImageObj, param: BinningParam) -> ImageObj:
    """Binning function on data with :py:func:`sigima.tools.image.binning`

    Args:
        src: input image object
        param: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(
        src,
        "binning",
        f"{param.sx}x{param.sy},{param.operation},"
        f"change_pixel_size={param.change_pixel_size}",
    )
    dst.data = sigima.tools.image.binning(
        src.data,
        sx=param.sx,
        sy=param.sy,
        operation=param.operation,
        dtype=None if param.dtype_str == "dtype" else param.dtype_str,
    )
    if param.change_pixel_size:
        if src.dx is not None and src.dy is not None:
            dst.dx = src.dx * param.sx
            dst.dy = src.dy * param.sy
    return dst


@computation_function()
def transpose(src: ImageObj) -> ImageObj:
    """Transpose image with :py:func:`numpy.transpose`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "transpose")
    dst.data = np.transpose(src.data)
    transformer.transform_roi(dst, "transpose")
    return dst
