# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image object definition
=======================

This module defines the main `ImageObj` class for representing 2D image data.

The `ImageObj` class provides:

- Data storage for 2D arrays with associated metadata
- Physical coordinate system with origin and pixel spacing
- Axis labeling and units
- Scale management (linear/logarithmic)
- DICOM template support
- ROI (Region of Interest) integration
- Coordinate conversion utilities (physical ↔ pixel)

This is the core class for image processing operations in Sigima.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Type

import guidata.dataset as gds
import numpy as np
from numpy import ma

from sigima.config import _
from sigima.objects import base
from sigima.objects.image.roi import ImageROI
from sigima.tools.datatypes import clip_astype


def to_builtin(obj) -> str | int | float | list | dict | np.ndarray | None:
    """Convert an object implementing a numeric value or collection
    into the corresponding builtin/NumPy type.

    Return None if conversion fails."""
    try:
        return int(obj) if int(obj) == float(obj) else float(obj)
    except (TypeError, ValueError):
        pass
    if isinstance(obj, str):
        return obj
    if hasattr(obj, "__iter__"):
        try:
            return list(obj)
        except (TypeError, ValueError):
            pass
    if hasattr(obj, "__dict__"):
        try:
            return dict(obj.__dict__)
        except (TypeError, ValueError):
            pass
    if isinstance(obj, np.ndarray):
        return obj
    return None


class ImageObj(gds.DataSet, base.BaseObj[ImageROI]):
    """Image object"""

    PREFIX = "i"
    VALID_DTYPES = (
        np.uint8,
        np.uint16,
        np.int16,
        np.int32,
        np.float32,
        np.float64,
        np.complex128,
    )

    def __init__(self, title=None, comment=None, icon=""):
        """Constructor

        Args:
            title: title
            comment: comment
            icon: icon
        """
        gds.DataSet.__init__(self, title, comment, icon)
        base.BaseObj.__init__(self)
        self._dicom_template = None

    @staticmethod
    def get_roi_class() -> Type[ImageROI]:
        """Return ROI class"""
        # Import here to avoid circular imports

        return ImageROI

    def __add_metadata(self, key: str, value: Any) -> None:
        """Add value to metadata if value can be converted into builtin/NumPy type

        Args:
            key: key
            value: value
        """
        stored_val = to_builtin(value)
        if stored_val is not None:
            self.metadata[key] = stored_val

    def __set_metadata_from(self, obj: Mapping | dict) -> None:
        """Set metadata from object: dict-like (only string keys are considered)
        or any other object (iterating over supported attributes)

        Args:
            obj: object
        """
        self.reset_metadata_to_defaults()
        ptn = r"__[\S_]*__$"
        if isinstance(obj, Mapping):
            for key, value in obj.items():
                if isinstance(key, str) and not re.match(ptn, key):
                    self.__add_metadata(key, value)
        else:
            for attrname in dir(obj):
                if attrname != "GroupLength" and not re.match(ptn, attrname):
                    try:
                        attr = getattr(obj, attrname)
                        if not callable(attr) and attr:
                            self.__add_metadata(attrname, attr)
                    except AttributeError:
                        pass

    @property
    def dicom_template(self):
        """Get DICOM template"""
        return self._dicom_template

    @dicom_template.setter
    def dicom_template(self, template):
        """Set DICOM template"""
        if template is not None:
            ipp = getattr(template, "ImagePositionPatient", None)
            if ipp is not None:
                self.x0, self.y0 = float(ipp[0]), float(ipp[1])
            pxs = getattr(template, "PixelSpacing", None)
            if pxs is not None:
                self.dy, self.dx = float(pxs[0]), float(pxs[1])
            self.__set_metadata_from(template)
            self._dicom_template = template

    _tabs = gds.BeginTabGroup("all")

    _datag = gds.BeginGroup(_("Data"))
    data = gds.FloatArrayItem(_("Data"))  # type: ignore[assignment]
    metadata = gds.DictItem(_("Metadata"), default={})  # type: ignore[assignment]
    annotations = gds.StringItem(_("Annotations"), default="").set_prop(
        "display",
        hide=True,
    )  # Annotations as a serialized JSON string  # type: ignore[assignment]
    _e_datag = gds.EndGroup(_("Data"))

    def _compute_xmin(self) -> float:
        """Compute Xmin"""
        if self.data is None or self.data.size == 0:
            return 0.0
        return self.x0

    def _compute_xmax(self) -> float:
        """Compute Xmax"""
        if self.data is None or self.data.size == 0:
            return 0.0
        return self.x0 + self.width - self.dx

    def _compute_ymin(self) -> float:
        """Compute Ymin"""
        if self.data is None or self.data.size == 0:
            return 0.0
        return self.y0

    def _compute_ymax(self) -> float:
        """Compute Ymax"""
        if self.data is None or self.data.size == 0:
            return 0.0
        return self.y0 + self.height - self.dy

    _dxdyg = gds.BeginGroup(f"{_('Origin')} / {_('Pixel spacing')}")
    _origin = gds.BeginGroup(_("Origin"))
    x0 = gds.FloatItem("X<sub>0</sub>", default=0.0)
    y0 = gds.FloatItem("Y<sub>0</sub>", default=0.0).set_pos(col=1)
    _e_origin = gds.EndGroup(_("Origin"))
    _pixel_spacing = gds.BeginGroup(_("Pixel spacing"))
    dx = gds.FloatItem("Δx", default=1.0, nonzero=True)
    dy = gds.FloatItem("Δy", default=1.0, nonzero=True).set_pos(col=1)
    _e_pixel_spacing = gds.EndGroup(_("Pixel spacing"))
    _boundaries = gds.BeginGroup(_("Extent"))
    xmin = gds.FloatItem("X<sub>MIN</sub>").set_computed(_compute_xmin)
    xmax = gds.FloatItem("X<sub>MAX</sub>").set_pos(col=1).set_computed(_compute_xmax)
    ymin = gds.FloatItem("Y<sub>MIN</sub>").set_computed(_compute_ymin)
    ymax = gds.FloatItem("Y<sub>MAX</sub>").set_pos(col=1).set_computed(_compute_ymax)
    _e_boundaries = gds.EndGroup(_("Extent"))
    _e_dxdyg = gds.EndGroup(f"{_('Origin')} / {_('Pixel spacing')}")

    _unitsg = gds.BeginGroup(_("Titles / Units"))
    title = gds.StringItem(_("Image title"), default=_("Untitled"))
    _tabs_u = gds.BeginTabGroup("units")
    _unitsx = gds.BeginGroup(_("X-axis"))
    xlabel = gds.StringItem(_("Title"), default="")
    xunit = gds.StringItem(_("Unit"), default="")
    _e_unitsx = gds.EndGroup(_("X-axis"))
    _unitsy = gds.BeginGroup(_("Y-axis"))
    ylabel = gds.StringItem(_("Title"), default="")
    yunit = gds.StringItem(_("Unit"), default="")
    _e_unitsy = gds.EndGroup(_("Y-axis"))
    _unitsz = gds.BeginGroup(_("Z-axis"))
    zlabel = gds.StringItem(_("Title"), default="")
    zunit = gds.StringItem(_("Unit"), default="")
    _e_unitsz = gds.EndGroup(_("Z-axis"))
    _e_tabs_u = gds.EndTabGroup("units")
    _e_unitsg = gds.EndGroup(_("Titles / Units"))

    _scalesg = gds.BeginGroup(_("Scales"))
    _prop_autoscale = gds.GetAttrProp("autoscale")
    autoscale = gds.BoolItem(_("Auto scale"), default=True).set_prop(
        "display", store=_prop_autoscale
    )
    _tabs_b = gds.BeginTabGroup("bounds")
    _boundsx = gds.BeginGroup(_("X-axis"))
    xscalelog = gds.BoolItem(_("Logarithmic scale"), default=False)
    xscalemin = gds.FloatItem(_("Lower bound"), check=False).set_prop(
        "display", active=gds.NotProp(_prop_autoscale)
    )
    xscalemax = gds.FloatItem(_("Upper bound"), check=False).set_prop(
        "display", active=gds.NotProp(_prop_autoscale)
    )
    _e_boundsx = gds.EndGroup(_("X-axis"))
    _boundsy = gds.BeginGroup(_("Y-axis"))
    yscalelog = gds.BoolItem(_("Logarithmic scale"), default=False)
    yscalemin = gds.FloatItem(_("Lower bound"), check=False).set_prop(
        "display", active=gds.NotProp(_prop_autoscale)
    )
    yscalemax = gds.FloatItem(_("Upper bound"), check=False).set_prop(
        "display", active=gds.NotProp(_prop_autoscale)
    )
    _e_boundsy = gds.EndGroup(_("Y-axis"))
    _boundsz = gds.BeginGroup(_("LUT range"))
    zscalemin = gds.FloatItem(_("Lower bound"), check=False)
    zscalemax = gds.FloatItem(_("Upper bound"), check=False)
    _e_boundsz = gds.EndGroup(_("LUT range"))
    _e_tabs_b = gds.EndTabGroup("bounds")
    _e_scalesg = gds.EndGroup(_("Scales"))

    _e_tabs = gds.EndTabGroup("all")

    @property
    def width(self) -> float:
        """Return image width, i.e. number of columns multiplied by pixel size"""
        return self.data.shape[1] * self.dx

    @property
    def height(self) -> float:
        """Return image height, i.e. number of rows multiplied by pixel size"""
        return self.data.shape[0] * self.dy

    @property
    def xc(self) -> float:
        """Return image center X-axis coordinate"""
        return self.x0 + 0.5 * self.width

    @property
    def yc(self) -> float:
        """Return image center Y-axis coordinate"""
        return self.y0 + 0.5 * self.height

    def get_data(self, roi_index: int | None = None) -> np.ndarray:
        """
        Return original data (if ROI is not defined or `roi_index` is None),
        or ROI data (if both ROI and `roi_index` are defined).

        Args:
            roi_index: ROI index

        Returns:
            Masked data
        """
        if self.roi is None or roi_index is None:
            view = self.data.view(ma.MaskedArray)
            view.mask = np.isnan(self.data)
            return view
        single_roi = self.roi.get_single_roi(roi_index)
        # pylint: disable=unbalanced-tuple-unpacking
        x0, y0, x1, y1 = self.physical_to_indices(single_roi.get_bounding_box(self))
        return self.get_masked_view()[y0:y1, x0:x1]

    def copy(
        self,
        title: str | None = None,
        dtype: np.dtype | None = None,
        all_metadata: bool = False,
    ) -> ImageObj:
        """Copy object.

        Args:
            title: title
            dtype: data type
            all_metadata: if True, copy all metadata, otherwise only basic metadata

        Returns:
            Copied object
        """
        title = self.title if title is None else title
        obj = ImageObj(title=title)
        obj.title = title
        obj.xlabel = self.xlabel
        obj.ylabel = self.ylabel
        obj.xunit = self.xunit
        obj.yunit = self.yunit
        obj.zunit = self.zunit
        obj.x0 = self.x0
        obj.y0 = self.y0
        obj.dx = self.dx
        obj.dy = self.dy
        obj.metadata = base.deepcopy_metadata(self.metadata, all_metadata=all_metadata)
        obj.annotations = self.annotations
        if self.data is not None:
            obj.data = np.array(self.data, copy=True, dtype=dtype)
        obj.dicom_template = self.dicom_template
        return obj

    def set_data_type(self, dtype: np.dtype) -> None:
        """Change data type.
        If data type is integer, clip values to the new data type's range, thus avoiding
        overflow or underflow.

        Args:
            Data type
        """
        self.data = clip_astype(self.data, dtype)

    def physical_to_indices(
        self, coords: list[float], clip: bool = False, as_float: bool = False
    ) -> list[int] | list[float]:
        """Convert coordinates from physical (real world) to indices (pixel)

        Args:
            coords: flat list of physical coordinates [x0, y0, x1, y1, ...]
            clip: if True, clip values to image boundaries
            as_float: if True, return float indices (i.e. without rounding)

        Returns:
            Indices

        Raises:
            ValueError: if coords does not contain an even number of elements
        """
        if len(coords) % 2 != 0:
            raise ValueError(
                "coords must contain an even number of elements (x, y pairs)."
            )
        indices = np.array(coords, float)
        if indices.size > 0:
            indices[::2] = (indices[::2] - self.x0) / self.dx
            indices[1::2] = (indices[1::2] - self.y0) / self.dy

        if clip:
            indices[::2] = np.clip(indices[::2], 0, self.data.shape[1] - 1)
            indices[1::2] = np.clip(indices[1::2], 0, self.data.shape[0] - 1)
        if as_float:
            return indices.tolist()
        return np.floor(indices + 0.5).astype(int).tolist()

    def indices_to_physical(self, indices: list[float]) -> list[int]:
        """Convert coordinates from indices to physical (real world)

        Args:
            indices: flat list of indices [x0, y0, x1, y1, ...]

        Returns:
            Coordinates

        Raises:
            ValueError: if indices does not contain an even number of elements
        """
        if len(indices) % 2 != 0:
            raise ValueError(
                "indices must contain an even number of elements (x, y pairs)."
            )
        coords = np.array(indices, float)
        if coords.size > 0:
            coords[::2] = coords[::2] * self.dx + self.x0
            coords[1::2] = coords[1::2] * self.dy + self.y0
        return coords.tolist()
