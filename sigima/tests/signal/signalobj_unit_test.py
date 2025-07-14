# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests around the `SignalObj` class and its creation from parameters.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

from collections.abc import Generator

import sigima.objects
from sigima.tests.env import execenv


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


if __name__ == "__main__":
    test_all_signal_types()
