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

from sigima.config import _
from sigima.enums import BorderMode, Interpolation2DMethod
from sigima.objects.image import ImageObj
from sigima.proc.base import dst_1_to_1
from sigima.proc.decorator import computation_function
from sigima.proc.transformations import transformer

# NOTE: Only parameter classes DEFINED in this module should be included in __all__.
# Parameter classes imported from other modules (like sigima.proc.base) should NOT
# be re-exported to avoid Sphinx cross-reference conflicts. The sigima.params module
# serves as the central API point that imports and re-exports all parameter classes.
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
    "Resampling2DParam",
    "resampling",
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

    prop = gds.ValueProp(False)

    angle = gds.FloatItem(f"{_('Angle')} (°)", default=0.0)
    mode = gds.ChoiceItem(_("Mode"), BorderMode, default=BorderMode.CONSTANT)
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

    prop = gds.ValueProp(False)

    zoom = gds.FloatItem(_("Zoom"), default=1.0)
    mode = gds.ChoiceItem(_("Mode"), BorderMode, default=BorderMode.CONSTANT)
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
    mode = p.mode
    dst = dst_1_to_1(src, "resize", f"zoom={p.zoom:.3f}")
    dst.data = spi.zoom(
        src.data,
        p.zoom,
        order=p.order,
        mode=mode,
        cval=p.cval,
        prefilter=p.prefilter,
    )
    if dst.dx is not None and dst.dy is not None:
        dst.dx, dst.dy = dst.dx / p.zoom, dst.dy / p.zoom
    return dst


@computation_function()
def transpose(src: ImageObj) -> ImageObj:
    """Transpose image with :py:func:`numpy.transpose`.

    Args:
        src: Input image object.

    Returns:
        Output image object.
    """
    dst = dst_1_to_1(src, "transpose")
    dst.data = np.transpose(src.data)
    dst.xlabel = src.ylabel
    dst.ylabel = src.xlabel
    dst.xunit = src.yunit
    dst.yunit = src.xunit
    dst.x0 = src.y0
    dst.y0 = src.x0
    dst.dx = src.dy
    dst.dy = src.dx
    transformer.transform_roi(dst, "transpose")
    return dst


class Resampling2DParam(gds.DataSet):
    """Resample parameters for 2D images"""

    # Output coordinate system
    xmin = gds.FloatItem(
        "X<sub>min</sub>",
        default=None,
        allow_none=True,
        help=_("Minimum X-coordinate of the output image"),
    )
    xmax = gds.FloatItem(
        "X<sub>max</sub>",
        default=None,
        allow_none=True,
        help=_("Maximum X-coordinate of the output image"),
    )
    ymin = gds.FloatItem(
        "Y<sub>min</sub>",
        default=None,
        allow_none=True,
        help=_("Minimum Y-coordinate of the output image"),
    )
    ymax = gds.FloatItem(
        "Y<sub>max</sub>",
        default=None,
        allow_none=True,
        help=_("Maximum Y-coordinate of the output image"),
    )

    # Mode selection
    _prop = gds.GetAttrProp("mode")
    _modes = (("dxy", _("Pixel size")), ("shape", _("Output shape")))
    mode = gds.ChoiceItem(_("Mode"), _modes, default="shape", radio=True).set_prop(
        "display", store=_prop
    )

    # Pixel size mode parameters
    dx = gds.FloatItem(
        "ΔX", default=None, allow_none=True, help=_("Pixel size in X direction")
    ).set_prop("display", active=gds.FuncProp(_prop, lambda x: x == "dxy"))
    dy = gds.FloatItem(
        "ΔY", default=None, allow_none=True, help=_("Pixel size in Y direction")
    ).set_prop("display", active=gds.FuncProp(_prop, lambda x: x == "dxy"))

    # Shape mode parameters
    width = gds.IntItem(
        _("Width"),
        default=None,
        allow_none=True,
        help=_("Output image width in pixels"),
    ).set_prop("display", active=gds.FuncProp(_prop, lambda x: x == "shape"))
    height = gds.IntItem(
        _("Height"),
        default=None,
        allow_none=True,
        help=_("Output image height in pixels"),
    ).set_prop("display", active=gds.FuncProp(_prop, lambda x: x == "shape"))

    # Interpolation parameters
    method = gds.ChoiceItem(
        _("Interpolation method"),
        Interpolation2DMethod,
        default=Interpolation2DMethod.LINEAR,
    )
    fill_value = gds.FloatItem(
        _("Fill value"),
        default=None,
        help=_(
            "Value to use for points outside the input image domain. "
            "If None, uses NaN for extrapolation."
        ),
        check=False,
    )

    def update_from_obj(self, obj: ImageObj) -> None:
        """Update parameters from an image object."""
        if self.xmin is None:
            self.xmin = obj.x0
        if self.xmax is None:
            self.xmax = obj.x0 + obj.width
        if self.ymin is None:
            self.ymin = obj.y0
        if self.ymax is None:
            self.ymax = obj.y0 + obj.height
        if self.dx is None:
            self.dx = obj.dx
        if self.dy is None:
            self.dy = obj.dy
        if self.width is None:
            self.width = obj.data.shape[1]
        if self.height is None:
            self.height = obj.data.shape[0]


@computation_function()
def resampling(src: ImageObj, p: Resampling2DParam) -> ImageObj:
    """Resample image to new coordinate grid using interpolation

    Args:
        src: source image
        p: resampling parameters

    Returns:
        Resampled image object
    """
    # Set output range - use source image bounds if not specified
    output_xmin = p.xmin if p.xmin is not None else src.x0
    output_xmax = p.xmax if p.xmax is not None else src.x0 + src.width
    output_ymin = p.ymin if p.ymin is not None else src.y0
    output_ymax = p.ymax if p.ymax is not None else src.y0 + src.height

    # Calculate output grid dimensions and spacing
    output_width_phys = output_xmax - output_xmin
    output_height_phys = output_ymax - output_ymin

    # Determine output grid parameters
    method: Interpolation2DMethod = p.method
    if p.mode == "dxy":
        # Calculate dimensions from pixel sizes
        if p.dx is None or p.dy is None:
            raise ValueError("dx and dy must be specified in pixel size mode")
        output_width = int(np.ceil(output_width_phys / p.dx))
        output_height = int(np.ceil(output_height_phys / p.dy))
        output_dx = p.dx
        output_dy = p.dy
        fill_suffix = f", fill_value={p.fill_value}" if p.fill_value is not None else ""
        suffix = f"method={method.value}, dx={p.dx:.3f}, dy={p.dy:.3f}{fill_suffix}"
    else:
        # Use specified shape
        if p.width is None or p.height is None:
            raise ValueError("width and height must be specified in shape mode")
        output_width = p.width
        output_height = p.height
        output_dx = output_width_phys / p.width if p.width > 0 else src.dx
        output_dy = output_height_phys / p.height if p.height > 0 else src.dy
        fill_suffix = f", fill_value={p.fill_value}" if p.fill_value is not None else ""
        suffix = f"method={method.value}, size=({p.width}x{p.height}){fill_suffix}"

    # Create destination image
    dst = dst_1_to_1(src, "resample", suffix)

    # Output coordinates (physical) - ensure we sample pixel centers, not boundaries
    # For an image spanning [xmin, xmax], we want to sample at pixel centers
    # The pixel centers should be distributed within the range,
    # not including the exact endpoints
    if output_width > 1:
        out_x = np.linspace(
            output_xmin + output_dx / 2, output_xmax - output_dx / 2, output_width
        )
    else:
        out_x = np.array([(output_xmin + output_xmax) / 2])

    if output_height > 1:
        out_y = np.linspace(
            output_ymin + output_dy / 2, output_ymax - output_dy / 2, output_height
        )
    else:
        out_y = np.array([(output_ymin + output_ymax) / 2])

    # Create meshgrids
    out_X, out_Y = np.meshgrid(out_x, out_y, indexing="xy")

    # Convert interpolation method to scipy parameter
    if method == Interpolation2DMethod.LINEAR:
        order = 1
    elif method == Interpolation2DMethod.CUBIC:
        order = 3
    elif method == Interpolation2DMethod.NEAREST:
        order = 0
    else:
        order = 1  # fallback to linear

    # Convert physical coordinates to source image indices
    src_i = (out_X - src.x0) / src.dx
    src_j = (out_Y - src.y0) / src.dy

    # Perform interpolation using map_coordinates
    # Note: map_coordinates expects (j, i) order (row, col)
    coordinates = np.array([src_j.ravel(), src_i.ravel()])

    # Determine fill value for interpolation
    cval = p.fill_value if p.fill_value is not None else np.nan

    # For NaN fill values, we need to work with float data to preserve NaN
    # Convert to float if necessary to allow NaN representation
    if np.isnan(cval) and not np.issubdtype(src.data.dtype, np.floating):
        input_data = src.data.astype(np.float64)
    else:
        input_data = src.data

    # Interpolate
    resampled_data = spi.map_coordinates(
        input_data, coordinates, order=order, mode="constant", cval=cval, prefilter=True
    ).reshape(output_height, output_width)

    # Set output data and coordinate system
    dst.data = resampled_data
    dst.x0 = output_xmin
    dst.y0 = output_ymin
    dst.dx = output_dx
    dst.dy = output_dy

    return dst
