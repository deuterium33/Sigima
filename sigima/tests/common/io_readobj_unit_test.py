# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
I/O test

Testing `sigima` specific formats.
"""

from __future__ import annotations

import numpy as np
import pytest

from sigima.io import read_images, read_signals
from sigima.io.ftlab import FTLabImageFile, imread_ftlabima, sigread_ftlabsig
from sigima.objects import ImageObj, SignalObj
from sigima.tests import guiutils, helpers
from sigima.tests.env import execenv


def __read_objs(fname: str) -> list[ImageObj] | list[SignalObj]:
    """Read objects from a file"""
    if "curve_formats" in fname:
        objs = read_signals(fname)
    else:
        objs = read_images(fname)
    for obj in objs:
        if np.all(np.isnan(obj.data)):
            raise ValueError("Data is all NaNs")
    for obj in objs:
        execenv.print(obj)
    return objs


def __read_and_view_objs(
    fname: str | None = None,
    title: str | None = None,
    request: pytest.FixtureRequest | None = None,
) -> list[ImageObj]:
    """Read and view objects from a file

    Args:
        fname: Name of the file to open.
        title: Title for the view.
        request: Pytest fixture request for GUI context.

    Returns:
        List of ImageObj or SignalObj read from the file.
    """
    guiutils.set_current_request(request)
    objs = __read_objs(fname)
    if guiutils.is_gui_enabled():
        from sigima.tests import vistools  # pylint: disable=import-outside-toplevel

        with vistools.qt_app_context():
            vistools.view_curves_and_images(objs, title=title)
    return objs


@helpers.try_open_test_data("Testing TXT file reader", "*.txt")
def test_open_txt(
    fname: str | None = None,
    title: str | None = None,
    request: pytest.FixtureRequest | None = None,
) -> None:
    """Testing TXT files"""
    __read_and_view_objs(fname, title, request)


@helpers.try_open_test_data("Testing CSV file reader", "*.csv")
def test_open_csv(
    fname: str | None = None,
    title: str | None = None,
    request: pytest.FixtureRequest | None = None,
) -> None:
    """Testing CSV files"""
    __read_and_view_objs(fname, title, request)


@helpers.try_open_test_data("Testing FTLab signal file reader", "*.sig")
def test_open_sigdata(
    fname: str | None = None,
    title: str | None = None,
    request: pytest.FixtureRequest | None = None,
) -> None:
    """Testing FTLab signal files"""
    __read_and_view_objs(fname, title, request)

    # Read the FTLab signal file and compare the data with the reference
    data = sigread_ftlabsig(fname)
    ref = read_signals(fname.replace(".sig", ".npy"))[0]
    helpers.check_array_result(f"{fname}", data, ref.xydata)


@helpers.try_open_test_data("Testing MAT-File reader", "*.mat")
def test_open_mat(
    fname: str | None = None,
    title: str | None = None,
    request: pytest.FixtureRequest | None = None,
) -> None:
    """Testing MAT files"""
    __read_and_view_objs(fname, title, request)


@helpers.try_open_test_data("Testing SIF file handler", "*.sif")
def test_open_sif(
    fname: str | None = None,
    title: str | None = None,
    request: pytest.FixtureRequest | None = None,
) -> None:
    """Testing SIF files"""
    __read_and_view_objs(fname, title, request)


@helpers.try_open_test_data("Testing SCOR-DATA file handler", "*.scor-data")
def test_open_scordata(
    fname: str | None = None,
    title: str | None = None,
    request: pytest.FixtureRequest | None = None,
) -> None:
    """Testing SCOR-DATA files"""
    __read_and_view_objs(fname, title, request)


@helpers.try_open_test_data("Testing FTLab image file handler", "*.ima")
def test_open_imadata(
    fname: str | None = None,
    title: str | None = None,
    request: pytest.FixtureRequest | None = None,
) -> None:
    """Testing FTLab image files."""
    __read_and_view_objs(fname, title, request)

    # Read the FTLab image file and show the data
    ftlab_file = FTLabImageFile(fname)
    ftlab_file.read()
    execenv.print(ftlab_file)

    # Read the FTLab image file and compare the data with the reference
    data = imread_ftlabima(fname)
    ref = read_images(fname.replace(".ima", ".npy"))[0]
    helpers.check_array_result(f"{fname}", data, ref.data)


@pytest.mark.gui
def test_read_obj_interactive() -> None:
    """Interactive test for I/O: read and view objects from various formats."""
    request = guiutils.DummyRequest(gui=True)
    test_open_txt(request=request)
    test_open_csv(request=request)
    test_open_sigdata(request=request)
    test_open_mat(request=request)
    test_open_sif(request=request)
    test_open_scordata(request=request)
    test_open_imadata(request=request)


if __name__ == "__main__":
    test_read_obj_interactive()
