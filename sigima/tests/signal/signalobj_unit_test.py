# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests around the `SignalObj` class and its creation from parameters.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

import os.path as osp

import numpy as np
import pytest

import sigima.io
import sigima.objects
from sigima.io.signal import SignalIORegistry
from sigima.tests import guiutils
from sigima.tests.data import iterate_signal_creation
from sigima.tests.env import execenv
from sigima.tests.helpers import (
    WorkdirRestoringTempDir,
    compare_metadata,
    read_test_objects,
)


# pylint: disable=unused-argument
def preprocess_signal_parameters(param: sigima.objects.NewSignalParam) -> None:
    """Preprocess signal parameters before creating the signal.

    Args:
        param: The signal parameters to preprocess.
    """
    # TODO: [P4] Add specific preprocessing for signal parameters if needed


def postprocess_signal_object(
    obj: sigima.objects.SignalObj, stype: sigima.objects.SignalTypes
) -> None:
    """Postprocess signal object after creation.

    Args:
        obj: The signal object to postprocess.
        stype: The type of the signal.
    """
    if stype == sigima.objects.SignalTypes.ZEROS:
        assert (obj.y == 0).all()


def test_all_signal_types() -> None:
    """Test all combinations of signal types and data sizes"""
    execenv.print(f"{test_all_signal_types.__doc__}:")
    for signal in iterate_signal_creation(
        preproc=preprocess_signal_parameters, postproc=postprocess_signal_object
    ):
        assert signal.x is not None and signal.y is not None
    execenv.print(f"{test_all_signal_types.__doc__}: OK")


def test_hdf5_signal_io() -> None:
    """Test HDF5 I/O for signal objects"""
    execenv.print(f"{test_hdf5_signal_io.__doc__}:")
    with WorkdirRestoringTempDir() as tmpdir:
        for fname, orig_signal in read_test_objects(SignalIORegistry):
            if orig_signal is None:
                execenv.print(f"  Skipping {fname} (not implemented)")
                continue
            # Save to HDF5
            filename = osp.join(tmpdir, f"test_{osp.basename(fname)}.h5sig")
            sigima.io.write_signal(filename, orig_signal)
            execenv.print(f"  Saved {filename}")
            # Read back
            fetch_signal = sigima.io.read_signal(filename)
            execenv.print(f"  Read {filename}")
            orig_x, orig_y = orig_signal.x, orig_signal.y
            orig_x: np.ndarray
            orig_y: np.ndarray
            x, y = fetch_signal.x, fetch_signal.y
            assert isinstance(x, np.ndarray)
            assert isinstance(y, np.ndarray)
            assert x.shape == orig_x.shape
            assert y.shape == orig_y.shape
            assert x.dtype == orig_x.dtype
            assert y.dtype == orig_y.dtype
            assert np.isclose(x, orig_x, atol=0.0).all()
            assert np.isclose(y, orig_y, atol=0.0).all()
            assert compare_metadata(fetch_signal.metadata, orig_signal.metadata.copy())
    execenv.print(f"{test_hdf5_signal_io.__doc__}: OK")


@pytest.mark.gui
def test_signal_parameters_interactive() -> None:
    """Test interactive creation of signal parameters"""
    execenv.print(f"{test_signal_parameters_interactive.__doc__}:")
    with guiutils.lazy_qt_app_context(force=True):
        for stype in sigima.objects.SignalTypes:
            param = sigima.objects.create_signal_parameters(stype)
            if isinstance(param, sigima.objects.CustomSignalParam):
                param.setup_array()
            if param.edit():
                execenv.print(f"  Edited parameters for {stype.value}:")
                execenv.print(f"    {param}")
            else:
                execenv.print(f"  Skipped editing parameters for {stype.value}")
    execenv.print(f"{test_signal_parameters_interactive.__doc__}: OK")


def test_create_signal() -> None:
    """Test creation of a signal object using `create_signal` function"""
    execenv.print(f"{test_create_signal.__doc__}:")
    # pylint: disable=import-outside-toplevel

    # Test all combinations of input parameters
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    dx = np.full_like(x, 0.1)
    dy = np.full_like(y, 0.01)
    metadata = {"source": "test", "description": "Test signal"}
    units = ("s", "V")
    labels = ("Time", "Amplitude")

    # 1. Create signal with all parameters
    title = "Some Signal"
    signal = sigima.objects.create_signal(
        title=title,
        x=x,
        y=y,
        dx=dx,
        dy=dy,
        metadata=metadata,
        units=units,
        labels=labels,
    )
    assert isinstance(signal, sigima.objects.SignalObj)
    assert signal.title == title
    assert np.array_equal(signal.x, x)
    assert np.array_equal(signal.y, y)
    assert np.array_equal(signal.dx, dx)
    assert np.array_equal(signal.dy, dy)
    assert signal.metadata == metadata
    assert (signal.xunit, signal.yunit) == units
    assert (signal.xlabel, signal.ylabel) == labels

    # 2. Create signal with only x and y
    signal = sigima.objects.create_signal("", x=x, y=y)
    assert isinstance(signal, sigima.objects.SignalObj)
    assert np.array_equal(signal.x, x)
    assert np.array_equal(signal.y, y)
    assert signal.dx is None
    assert signal.dy is None
    assert signal.metadata == {}
    assert (signal.xunit, signal.yunit) == ("", "")
    assert (signal.xlabel, signal.ylabel) == ("", "")

    # 3. Create signal with only x, y, and dx
    signal = sigima.objects.create_signal("", x=x, y=y, dx=dx)
    assert isinstance(signal, sigima.objects.SignalObj)
    assert np.array_equal(signal.x, x)
    assert np.array_equal(signal.y, y)
    assert np.array_equal(signal.dx, dx)
    assert signal.dy is None

    # 4. Create signal with only x, y, and dy
    signal = sigima.objects.create_signal("", x=x, y=y, dy=dy)
    assert isinstance(signal, sigima.objects.SignalObj)
    assert np.array_equal(signal.x, x)
    assert np.array_equal(signal.y, y)
    assert signal.dx is None
    assert np.array_equal(signal.dy, dy)

    execenv.print(f"{test_create_signal.__doc__}: OK")


if __name__ == "__main__":
    test_signal_parameters_interactive()
    test_all_signal_types()
    test_hdf5_signal_io()
