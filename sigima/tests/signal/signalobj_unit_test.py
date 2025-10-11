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
    # Add here specific preprocessing for signal parameters if needed


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
    assert not signal.metadata
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


def test_create_signal_from_param() -> None:
    """Test creation of a signal object using `create_signal_from_param` function"""
    execenv.print(f"{test_create_signal_from_param.__doc__}:")

    # Test with different signal parameter types
    test_cases = [
        # Basic periodic functions
        (sigima.objects.SinusParam, "sinus"),
        (sigima.objects.CosinusParam, "cosinus"),
        (sigima.objects.SawtoothParam, "sawtooth"),
        (sigima.objects.TriangleParam, "triangle"),
        (sigima.objects.SquareParam, "square"),
        (sigima.objects.SincParam, "sinc"),
        # Mathematical functions
        (sigima.objects.GaussParam, "gaussian"),
        (sigima.objects.LorentzParam, "lorentzian"),
        (sigima.objects.ExponentialParam, "exponential"),
        (sigima.objects.LogisticParam, "logistic"),
        (sigima.objects.LinearChirpParam, "linear_chirp"),
        (sigima.objects.StepParam, "step"),
        (sigima.objects.PulseParam, "pulse"),
        (sigima.objects.SquarePulseParam, "square_pulse"),
        (sigima.objects.StepPulseParam, "step_pulse"),
        (sigima.objects.PolyParam, "polynomial"),
        # Noise and random signals
        (sigima.objects.NormalDistribution1DParam, "normal_noise"),
        (sigima.objects.PoissonDistribution1DParam, "poisson_noise"),
        (sigima.objects.UniformDistribution1DParam, "uniform_noise"),
        (sigima.objects.ZerosParam, "zeros"),
        # Other signals
        (sigima.objects.CustomSignalParam, "custom"),
        (sigima.objects.VoigtParam, "voigt"),
        (sigima.objects.PlanckParam, "planck"),
    ]

    # Raise an exception if sigima.objects.signal contain *Param classes not listed here
    param_classes = dict(test_cases)
    for attr_name in dir(sigima.objects):
        attr = getattr(sigima.objects, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, sigima.objects.NewSignalParam)
            and attr is not sigima.objects.NewSignalParam
            and attr is not sigima.objects.CustomSignalParam
            and attr not in param_classes
        ):
            raise AssertionError(f"Missing test case for {attr.__name__}")

    for param_class, name in test_cases:
        # Create parameter instance with default values
        param = param_class.create(size=100, xmin=1.0, xmax=10.0)
        param.title = f"Test {name} signal"

        # Test the function
        signal = sigima.objects.create_signal_from_param(param)

        # Verify the returned object
        assert isinstance(signal, sigima.objects.SignalObj), (
            f"Expected SignalObj, got {type(signal)} for {name}"
        )
        assert signal.title == f"Test {name} signal", (
            f"Title mismatch for {name}: expected 'Test {name} signal', "
            f"got '{signal.title}'"
        )
        assert signal.x is not None, f"X data is None for {name}"
        assert signal.y is not None, f"Y data is None for {name}"
        assert len(signal.x) == 100, f"X length mismatch for {name}"
        assert len(signal.y) == 100, f"Y length mismatch for {name}"
        assert isinstance(signal.x, np.ndarray), f"X is not ndarray for {name}"
        assert isinstance(signal.y, np.ndarray), f"Y is not ndarray for {name}"

        execenv.print(f"  Created {name} signal: OK")

    # Test with custom parameters and title generation
    param = sigima.objects.GaussParam.create(size=50, xmin=-5.0, xmax=5.0)
    param.title = ""  # Empty title should trigger automatic numbering
    signal = sigima.objects.create_signal_from_param(param)

    assert signal.title != "", "Empty title should be replaced"

    # Test parameter validation with units and labels
    param = sigima.objects.SinusParam()
    param.title = "Sine wave test"
    # xunit is set by default to "s" in SinusParam
    assert param.xunit == "s"
    param.yunit = "V"
    param.xlabel = "Time"
    param.ylabel = "Amplitude"

    signal = sigima.objects.create_signal_from_param(param)

    expected_xunit = "s"
    assert signal.xunit == expected_xunit, (
        f"X unit mismatch: expected '{expected_xunit}', got '{signal.xunit}'"
    )
    expected_yunit = "V"
    assert signal.yunit == expected_yunit, (
        f"Y unit mismatch: expected '{expected_yunit}', got '{signal.yunit}'"
    )
    expected_xlabel = "Time"
    assert signal.xlabel == expected_xlabel, (
        f"X label mismatch: expected '{expected_xlabel}', got '{signal.xlabel}'"
    )
    expected_ylabel = "Amplitude"
    assert signal.ylabel == expected_ylabel, (
        f"Y label mismatch: expected '{expected_ylabel}', got '{signal.ylabel}'"
    )

    execenv.print(f"{test_create_signal_from_param.__doc__}: OK")


def test_signal_copy() -> None:
    """Test copying signal objects with all attributes"""
    execenv.print(f"{test_signal_copy.__doc__}:")

    # Create a base signal with some data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    dx = np.full_like(x, 0.1)
    dy = np.full_like(y, 0.01)
    title = "Original Signal"
    metadata = {"key1": "value1", "key2": 42}
    units = ("s", "V")
    labels = ("Time", "Voltage")

    # Test 1: Copy signal with all attributes
    execenv.print("  Test 1: Copy signal with all attributes")
    signal = sigima.objects.create_signal(
        title=title,
        x=x,
        y=y,
        dx=dx,
        dy=dy,
        metadata=metadata.copy(),
        units=units,
        labels=labels,
    )

    # Set scale attributes
    signal.autoscale = False
    signal.xscalelog = True
    signal.xscalemin = 1.0
    signal.xscalemax = 9.0
    signal.yscalelog = False
    signal.yscalemin = -1.5
    signal.yscalemax = 1.5

    # Copy the signal
    copied = signal.copy()

    # Verify the copy
    assert copied is not signal
    assert copied.title == signal.title
    assert np.array_equal(copied.x, signal.x)
    assert np.array_equal(copied.y, signal.y)
    assert np.array_equal(copied.dx, signal.dx)
    assert np.array_equal(copied.dy, signal.dy)
    assert copied.xydata is not signal.xydata  # Different array objects
    assert copied.metadata == signal.metadata
    assert copied.metadata is not signal.metadata
    assert (copied.xunit, copied.yunit) == units
    assert (copied.xlabel, copied.ylabel) == labels

    # Verify scale attributes are preserved
    assert copied.autoscale == signal.autoscale
    assert copied.xscalelog == signal.xscalelog
    assert copied.xscalemin == signal.xscalemin
    assert copied.xscalemax == signal.xscalemax
    assert copied.yscalelog == signal.yscalelog
    assert copied.yscalemin == signal.yscalemin
    assert copied.yscalemax == signal.yscalemax
    execenv.print("    ✓ All attributes correctly copied")

    # Test 2: Copy with title override
    execenv.print("  Test 2: Copy with custom title")
    new_title = "Copied Signal"
    copied_with_title = signal.copy(title=new_title)
    assert copied_with_title.title == new_title
    assert copied_with_title.autoscale == signal.autoscale
    assert np.array_equal(copied_with_title.x, signal.x)
    execenv.print("    ✓ Title override works correctly")

    # Test 3: Copy with metadata filtering
    execenv.print("  Test 3: Copy with metadata filtering")
    copied_basic_meta = signal.copy(all_metadata=False)
    assert copied_basic_meta.autoscale == signal.autoscale
    assert copied_basic_meta.xscalelog == signal.xscalelog
    execenv.print("    ✓ Metadata filtering works correctly")

    # Test 4: Copy signal without error bars
    execenv.print("  Test 4: Copy signal without error bars")
    signal_no_err = sigima.objects.create_signal(
        title="Signal without error bars",
        x=x,
        y=y,
        units=units,
        labels=labels,
    )
    signal_no_err.autoscale = True
    signal_no_err.yscalelog = True

    copied_no_err = signal_no_err.copy()
    assert copied_no_err.dx is None
    assert copied_no_err.dy is None
    assert copied_no_err.autoscale is True
    assert copied_no_err.yscalelog is True
    execenv.print("    ✓ Signal without error bars copied correctly")

    execenv.print(f"{test_signal_copy.__doc__}: OK")


if __name__ == "__main__":
    test_signal_parameters_interactive()
    test_all_signal_types()
    test_hdf5_signal_io()
    test_create_signal()
    test_create_signal_from_param()
    test_signal_copy()
