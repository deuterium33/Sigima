# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests around the `SignalObj` class and its creation from parameters.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

import os.path as osp
from collections.abc import Generator

import numpy as np

import sigima.io
import sigima.objects
from sigima.io.signal import SignalIORegistry
from sigima.tests.env import execenv
from sigima.tests.helpers import (
    WorkdirRestoringTempDir,
    compare_metadata,
    read_test_objects,
)


def iterate_signal_creation(
    data_size: int = 500, non_zero: bool = False, verbose: bool = True
) -> Generator[sigima.objects.SignalObj, None, None]:
    """Iterate over all possible signals created from parameters"""
    if verbose:
        execenv.print(
            f"  Iterating over {len(sigima.objects.SignalTypes)} signal types "
            f"(size={data_size}, non_zero={non_zero}):"
        )
    for stype in sigima.objects.SignalTypes:
        if non_zero and stype in (sigima.objects.SignalTypes.ZEROS,):
            continue
        if verbose:
            execenv.print(f"    {stype.value}")
        base_param = sigima.objects.NewSignalParam.create(stype=stype, size=data_size)
        if stype == sigima.objects.SignalTypes.UNIFORMRANDOM:
            extra_param = sigima.objects.UniformRandomParam()
        elif stype == sigima.objects.SignalTypes.NORMALRANDOM:
            extra_param = sigima.objects.NormalRandomParam()
        elif stype in (
            sigima.objects.SignalTypes.GAUSS,
            sigima.objects.SignalTypes.LORENTZ,
            sigima.objects.SignalTypes.VOIGT,
        ):
            extra_param = sigima.objects.GaussLorentzVoigtParam()
        elif stype in (
            sigima.objects.SignalTypes.SINUS,
            sigima.objects.SignalTypes.COSINUS,
            sigima.objects.SignalTypes.SAWTOOTH,
            sigima.objects.SignalTypes.TRIANGLE,
            sigima.objects.SignalTypes.SQUARE,
            sigima.objects.SignalTypes.SINC,
        ):
            extra_param = sigima.objects.PeriodicParam()
        elif stype == sigima.objects.SignalTypes.STEP:
            extra_param = sigima.objects.StepParam()
        elif stype == sigima.objects.SignalTypes.EXPONENTIAL:
            extra_param = sigima.objects.ExponentialParam()
        elif stype == sigima.objects.SignalTypes.PULSE:
            extra_param = sigima.objects.PulseParam()
        elif stype == sigima.objects.SignalTypes.POLYNOMIAL:
            extra_param = sigima.objects.PolyParam()
        elif stype == sigima.objects.SignalTypes.EXPERIMENTAL:
            extra_param = sigima.objects.ExperimentalSignalParam()
        else:
            extra_param = None
        signal = sigima.objects.create_signal_from_param(
            base_param, extra_param=extra_param
        )
        if stype == sigima.objects.SignalTypes.ZEROS:
            assert (signal.y == 0).all()
        yield signal


def test_all_signal_types() -> None:
    """Test all combinations of signal types and data sizes"""
    execenv.print(f"{test_all_signal_types.__doc__}:")
    for signal in iterate_signal_creation():
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
            filename = osp.join(tmpdir, f"test_{osp.basename(fname)}.h5")
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


if __name__ == "__main__":
    test_all_signal_types()
    test_hdf5_signal_io()
