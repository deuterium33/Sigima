# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Signal ROI utilities
====================

This module provides Region of Interest (ROI) classes and utilities for signal objects.

The module includes:

- `ROI1DParam`: Parameter class for 1D signal ROIs
- `SegmentROI`: Single ROI representing a segment of a signal
- `SignalROI`: Collection of signal ROIs with operations
- `create_signal_roi`: Factory function for creating signal ROI objects

These classes enable defining and working with regions of interest in 1D signal data,
supporting operations like data extraction, masking, and parameter conversion.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

from typing import TYPE_CHECKING, Type

import guidata.dataset as gds
import numpy as np

from sigima.config import _
from sigima.objects import base

if TYPE_CHECKING:
    from sigima.objects.signal.object import SignalObj


class ROI1DParam(base.BaseROIParam["SignalObj", "SegmentROI"]):
    """Signal ROI parameters"""

    # Note: in this class, the ROI parameters are stored as X coordinates

    title = gds.StringItem(_("ROI title"), default="")
    xmin = gds.FloatItem(_("First point coordinate"), default=0.0)
    xmax = gds.FloatItem(_("Last point coordinate"), default=1.0)

    def to_single_roi(self, obj: SignalObj) -> SegmentROI:
        """Convert parameters to single ROI

        Args:
            obj: signal object

        Returns:
            Single ROI
        """
        assert isinstance(self.xmin, float) and isinstance(self.xmax, float)
        return SegmentROI([self.xmin, self.xmax], False, title=self.title)

    def get_data(self, obj: SignalObj) -> np.ndarray:
        """Get signal data in ROI

        Args:
            obj: signal object

        Returns:
            Data in ROI
        """
        assert isinstance(self.xmin, float) and isinstance(self.xmax, float)
        imin, imax = np.searchsorted(obj.x, [self.xmin, self.xmax])
        return np.array([obj.x[imin:imax], obj.y[imin:imax]])


class SegmentROI(base.BaseSingleROI["SignalObj", ROI1DParam]):
    """Segment ROI

    Args:
        coords: ROI coordinates (xmin, xmax)
        title: ROI title
    """

    # Note: in this class, the ROI parameters are stored as X indices

    def check_coords(self) -> None:
        """Check if coords are valid

        Raises:
            ValueError: invalid coords
        """
        if len(self.coords) != 2:
            raise ValueError("Invalid ROI segment coords (2 values expected)")
        if self.coords[0] >= self.coords[1]:
            raise ValueError("Invalid ROI segment coords (xmin >= xmax)")

    def get_data(self, obj: SignalObj) -> tuple[np.ndarray, np.ndarray]:
        """Get signal data in ROI

        Args:
            obj: signal object

        Returns:
            Data in ROI
        """
        imin, imax = self.get_indices_coords(obj)
        return obj.x[imin:imax], obj.y[imin:imax]

    def to_mask(self, obj: SignalObj) -> np.ndarray:
        """Create mask from ROI

        Args:
            obj: signal object

        Returns:
            Mask (boolean array where True values are inside the ROI)
        """
        mask = np.ones_like(obj.xydata, dtype=bool)
        imin, imax = self.get_indices_coords(obj)
        mask[:, imin:imax] = False
        return mask

    # pylint: disable=unused-argument
    def to_param(self, obj: SignalObj, index: int) -> ROI1DParam:
        """Convert ROI to parameters

        Args:
            obj: object (signal), for physical-indices coordinates conversion
            index: ROI index
        """
        gtitle = base.get_generic_roi_title(index)
        param = ROI1DParam(gtitle)
        param.title = self.title or gtitle
        param.xmin, param.xmax = self.get_physical_coords(obj)
        return param


class SignalROI(base.BaseROI["SignalObj", SegmentROI, ROI1DParam]):
    """Signal Regions of Interest

    Args:
        inverse: if True, ROI is outside the region
    """

    PREFIX = "s"

    @staticmethod
    def get_compatible_single_roi_classes() -> list[Type[SegmentROI]]:
        """Return compatible single ROI classes"""
        return [SegmentROI]

    def to_mask(self, obj: SignalObj) -> np.ndarray:
        """Create mask from ROI

        Args:
            obj: signal object

        Returns:
            Mask (boolean array where True values are inside the ROI)
        """
        mask = np.ones_like(obj.xydata, dtype=bool)
        if self.single_rois:
            for roi in self.single_rois:
                mask &= roi.to_mask(obj)
        else:
            # If no single ROIs, the mask is empty (no ROI defined)
            mask[:] = False
        return mask


def create_signal_roi(
    coords: np.ndarray | list[float] | list[list[float]],
    indices: bool = False,
    inverse: bool = False,
    title: str = "",
) -> SignalROI:
    """Create Signal Regions of Interest (ROI) object.
    More ROIs can be added to the object after creation, using the `add_roi` method.

    Args:
        coords: single ROI coordinates `[xmin, xmax]`, or multiple ROIs coordinates
         `[[xmin1, xmax1], [xmin2, xmax2], ...]` (lists or NumPy arrays)
        indices: if True, coordinates are indices, if False, they are physical values
         (default to False for signals)
        inverse: if True, ROI is outside the region
        title: title

    Returns:
        Regions of Interest (ROI) object

    Raises:
        ValueError: if the number of coordinates is not even
    """
    coords = np.array(coords, float)
    if coords.ndim == 1:
        coords = coords.reshape(1, -1)
    roi = SignalROI(inverse)
    for row in coords:
        roi.add_roi(SegmentROI(row, indices=indices, title=title))
    return roi
