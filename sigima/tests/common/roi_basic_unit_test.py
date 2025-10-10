# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
ROI basic unit tests
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

import sigima.objects
from sigima.tests.data import (
    create_multigaussian_image,
    create_paracetamol_signal,
    create_test_image_rois,
    create_test_signal_rois,
)
from sigima.tests.env import execenv


def __conversion_methods(
    roi: sigima.objects.SignalROI | sigima.objects.ImageROI,
    obj: sigima.objects.SignalObj | sigima.objects.ImageObj,
) -> None:
    """Test conversion methods for single ROI objects"""
    execenv.print("    test `to_dict` and `from_dict` methods")
    roi_dict = roi.to_dict()
    roi_new = obj.get_roi_class().from_dict(roi_dict)
    assert roi.get_single_roi(0) == roi_new.get_single_roi(0)


def test_signal_roi_creation() -> None:
    """Test signal ROI creation and conversion methods"""
    obj = create_paracetamol_signal()
    for roi in create_test_signal_rois(obj):
        __conversion_methods(roi, obj)


def test_image_roi_creation() -> None:
    """Test image ROI creation and conversion methods"""
    obj = create_multigaussian_image()
    # Update to use new coordinate API instead of setting dx/dy directly
    if obj.data is not None:
        obj.set_uniform_coords(0.035, 0.035)
    for roi in create_test_image_rois(obj):
        __conversion_methods(roi, obj)


def test_image_roi_modification() -> None:
    """Test image ROI modification methods"""
    obj = create_multigaussian_image()
    roi = list(create_test_image_rois(obj))[0]

    # Set image's ROI
    obj.roi = roi
    assert obj.roi == roi
    nb_single_rois = len(roi.single_rois)

    # Modify the ROI directly from the image's ROI attribute
    # (for example, we try to remove a single ROI)
    old_single_roi = obj.roi.single_rois.pop(0)
    assert len(obj.roi.single_rois) == nb_single_rois - 1

    # Add it back
    obj.roi.add_roi(old_single_roi)
    assert len(obj.roi.single_rois) == nb_single_rois


if __name__ == "__main__":
    test_signal_roi_creation()
    test_image_roi_creation()
    test_image_roi_modification()
