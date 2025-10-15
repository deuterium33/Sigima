# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
I/O metadata and ROI functions
"""

from __future__ import annotations

import os.path as osp

from sigima.io.common.objmeta import (
    read_metadata,
    read_roi,
    write_metadata,
    write_roi,
)
from sigima.tests.data import (
    create_multigaussian_image,
    create_paracetamol_signal,
    create_test_image_rois,
    create_test_metadata,
    create_test_signal_rois,
)
from sigima.tests.env import execenv
from sigima.tests.helpers import WorkdirRestoringTempDir, compare_metadata


def test_signal_roi_io():
    """Test reading and writing of signal ROIs."""
    execenv.print("==============================================")
    execenv.print("Testing signal ROI I/O")
    execenv.print("==============================================")
    with WorkdirRestoringTempDir() as temp_dir:
        obj = create_paracetamol_signal()
        for orig_roi in create_test_signal_rois(obj):
            fname = osp.join(temp_dir, "test_signal_roi.json")
            write_roi(fname, orig_roi)
            roi = read_roi(fname)
            try:
                compare_metadata(roi.to_dict(), orig_roi.to_dict(), raise_on_diff=True)
            except AssertionError as exc:
                raise AssertionError(
                    "Signal ROI read from file does not match original"
                ) from exc


def test_image_roi_io():
    """Test reading and writing of image ROIs."""
    execenv.print("==============================================")
    execenv.print("Testing image ROI I/O")
    execenv.print("==============================================")
    with WorkdirRestoringTempDir() as temp_dir:
        obj = create_multigaussian_image()
        for orig_roi in create_test_image_rois(obj):
            fname = osp.join(temp_dir, "test_image_roi.json")
            write_roi(fname, orig_roi)
            roi = read_roi(fname)
            try:
                compare_metadata(roi.to_dict(), orig_roi.to_dict(), raise_on_diff=True)
            except AssertionError as exc:
                raise AssertionError(
                    "Image ROI read from file does not match original"
                ) from exc


def test_metadata_io():
    """Test reading and writing of metadata."""
    execenv.print("==============================================")
    execenv.print("Testing metadata I/O")
    execenv.print("==============================================")
    with WorkdirRestoringTempDir() as temp_dir:
        orig_metadata = create_test_metadata()
        fname = osp.join(temp_dir, "test_metadata.json")
        write_metadata(fname, orig_metadata)
        metadata = read_metadata(fname)
        try:
            compare_metadata(metadata, orig_metadata, raise_on_diff=True)
        except AssertionError as exc:
            raise AssertionError(
                "Metadata read from file does not match original"
            ) from exc


if __name__ == "__main__":
    test_signal_roi_io()
    test_image_roi_io()
    test_metadata_io()
