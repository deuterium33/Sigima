# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for signal operations
--------------------------------

Features from the "Operations" menu are covered by this test.
The "Operations" menu contains basic operations on signals, such as
addition, multiplication, division, and more.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import warnings

import numpy as np
import pytest

import sigima.objects
import sigima.params
import sigima.proc.signal
import sigima.tests.data
from sigima.objects.signal import SignalObj
from sigima.proc.base import AngleUnit, AngleUnitParam
from sigima.proc.signal import complex_from_magnitude_phase, complex_from_real_imag
from sigima.tests.helpers import check_array_result
from sigima.tools.coordinates import polar_to_complex


def __create_two_signals() -> tuple[sigima.objects.SignalObj, sigima.objects.SignalObj]:
    """Create two signals for testing."""
    s1 = sigima.tests.data.create_periodic_signal(
        sigima.objects.SignalTypes.COSINUS, freq=50.0, size=100
    )
    s2 = sigima.tests.data.create_periodic_signal(
        sigima.objects.SignalTypes.SINUS, freq=25.0, size=100
    )
    return s1, s2


def __create_n_signals(n: int = 100) -> list[sigima.objects.SignalObj]:
    """Create a list of N different signals for testing."""
    signals = []
    for i in range(n):
        s = sigima.tests.data.create_periodic_signal(
            sigima.objects.SignalTypes.COSINUS,
            freq=50.0 + i,
            size=100,
            a=(i + 1) * 0.1,
        )
        signals.append(s)
    return signals


def __create_one_signal_and_constant() -> tuple[
    sigima.objects.SignalObj, sigima.params.ConstantParam
]:
    """Create one signal and a constant for testing."""
    s1 = sigima.tests.data.create_periodic_signal(
        sigima.objects.SignalTypes.COSINUS, freq=50.0, size=100
    )
    param = sigima.params.ConstantParam.create(value=-np.pi)
    return s1, param


@pytest.mark.validation
def test_signal_addition() -> None:
    """Signal addition test."""
    slist = __create_n_signals()
    n = len(slist)
    s3 = sigima.proc.signal.addition(slist)
    res = s3.y
    exp = np.zeros_like(s3.y)
    for s in slist:
        exp += s.y
    check_array_result(f"Addition of {n} signals", res, exp)


@pytest.mark.validation
def test_signal_average() -> None:
    """Signal average test."""
    slist = __create_n_signals()
    n = len(slist)
    s3 = sigima.proc.signal.average(slist)
    res = s3.y
    exp = np.zeros_like(s3.y)
    for s in slist:
        exp += s.y
    exp /= n
    check_array_result(f"Average of {n} signals", res, exp)


@pytest.mark.validation
def test_signal_product() -> None:
    """Signal multiplication test."""
    slist = __create_n_signals()
    n = len(slist)
    s3 = sigima.proc.signal.product(slist)
    res = s3.y
    exp = np.ones_like(s3.y)
    for s in slist:
        exp *= s.y
    check_array_result(f"Product of {n} signals", res, exp)


@pytest.mark.validation
def test_signal_difference() -> None:
    """Signal difference test."""
    s1, s2 = __create_two_signals()
    s3 = sigima.proc.signal.difference(s1, s2)
    check_array_result("Signal difference", s3.y, s1.y - s2.y)


@pytest.mark.validation
def test_signal_quadratic_difference() -> None:
    """Signal quadratic difference validation test."""
    s1, s2 = __create_two_signals()
    s3 = sigima.proc.signal.quadratic_difference(s1, s2)
    check_array_result("Signal quadratic difference", s3.y, (s1.y - s2.y) / np.sqrt(2))


@pytest.mark.validation
def test_signal_division() -> None:
    """Signal division test."""
    s1, s2 = __create_two_signals()
    s3 = sigima.proc.signal.division(s1, s2)
    check_array_result("Signal division", s3.y, s1.y / s2.y)


@pytest.mark.validation
def test_signal_addition_constant() -> None:
    """Signal addition with constant test."""
    s1, param = __create_one_signal_and_constant()
    s2 = sigima.proc.signal.addition_constant(s1, param)
    check_array_result("Signal addition with constant", s2.y, s1.y + param.value)


@pytest.mark.validation
def test_signal_product_constant() -> None:
    """Signal multiplication by constant test."""
    s1, param = __create_one_signal_and_constant()
    s2 = sigima.proc.signal.product_constant(s1, param)
    check_array_result("Signal multiplication by constant", s2.y, s1.y * param.value)


@pytest.mark.validation
def test_signal_difference_constant() -> None:
    """Signal difference with constant test."""
    s1, param = __create_one_signal_and_constant()
    s2 = sigima.proc.signal.difference_constant(s1, param)
    check_array_result("Signal difference with constant", s2.y, s1.y - param.value)


@pytest.mark.validation
def test_signal_division_constant() -> None:
    """Signal division by constant test."""
    s1, param = __create_one_signal_and_constant()
    s2 = sigima.proc.signal.division_constant(s1, param)
    check_array_result("Signal division by constant", s2.y, s1.y / param.value)


@pytest.mark.validation
def test_signal_inverse() -> None:
    """Signal inversion validation test."""
    s1 = __create_two_signals()[0]
    inv_signal = sigima.proc.signal.inverse(s1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        exp = 1.0 / s1.y
        exp[np.isinf(exp)] = np.nan
    check_array_result("Signal inverse", inv_signal.y, exp)


@pytest.mark.validation
def test_signal_absolute() -> None:
    """Absolute value validation test."""
    s1 = __create_two_signals()[0]
    abs_signal = sigima.proc.signal.absolute(s1)
    check_array_result("Absolute value", abs_signal.y, np.abs(s1.y))


@pytest.mark.validation
def test_signal_real() -> None:
    """Real part validation test."""
    s1 = __create_two_signals()[0]
    re_signal = sigima.proc.signal.real(s1)
    check_array_result("Real part", re_signal.y, np.real(s1.y))


@pytest.mark.validation
def test_signal_imag() -> None:
    """Imaginary part validation test."""
    s1 = __create_two_signals()[0]
    im_signal = sigima.proc.signal.imag(s1)
    check_array_result("Imaginary part", im_signal.y, np.imag(s1.y))


@pytest.mark.validation
def test_signal_complex_from_real_imag() -> None:
    """Test :py:func:`sigima.proc.signal.complex_from_real_imag`."""
    x = np.linspace(0.0, 1.0, 5)
    real = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    imag = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    # Create SignalObj instances for real and imaginary parts
    s_real = SignalObj("real")
    s_real.set_xydata(x, real)
    s_imag = SignalObj("imag")
    s_imag.set_xydata(x, imag)
    # Create complex signal from real and imaginary parts
    result = complex_from_real_imag(s_real, s_imag)
    check_array_result(
        "complex_from_real_imag",
        result.y,
        real + 1j * imag,
    )


@pytest.mark.validation
def test_signal_phase() -> None:
    """Phase angle validation test."""
    s1 = __create_two_signals()[0]
    # Make a complex signal for testing
    y_complex = s1.y + 1j * s1.y[::-1]
    s_complex = sigima.objects.create_signal("complex", s1.x, y_complex)

    # Test output in radians, no unwrapping
    p_rad = sigima.params.PhaseParam.create(unit=AngleUnit.radian, unwrap=False)
    phase_signal_rad = sigima.proc.signal.phase(s_complex, p_rad)
    check_array_result("Phase|rad", phase_signal_rad.y, np.angle(y_complex))
    # Test output in degrees, no unwrapping
    p_deg = sigima.params.PhaseParam.create(unit=AngleUnit.degree, unwrap=False)
    phase_signal_deg = sigima.proc.signal.phase(s_complex, p_deg)
    check_array_result("Phase|deg", phase_signal_deg.y, np.angle(y_complex, deg=True))
    # Test output in radians, with unwrapping
    p_rad_unwrap = sigima.params.PhaseParam.create(unit=AngleUnit.radian, unwrap=True)
    phase_signal_rad_unwrap = sigima.proc.signal.phase(s_complex, p_rad_unwrap)
    check_array_result(
        "Phase|unwrapping|rad",
        phase_signal_rad_unwrap.y,
        np.unwrap(np.angle(y_complex)),
    )
    # Test output in degrees, with unwrapping
    p_deg_unwrap = sigima.params.PhaseParam.create(unit=AngleUnit.degree, unwrap=True)
    phase_signal_deg_unwrap = sigima.proc.signal.phase(s_complex, p_deg_unwrap)
    check_array_result(
        "Phase|unwrapping|deg",
        phase_signal_deg_unwrap.y,
        np.unwrap(np.angle(y_complex, deg=True), period=360.0),
    )


complex_from_magnitude_phase_parameters = [
    (np.array([0.0, np.pi / 2, np.pi, 3.0 * np.pi / 2.0, 0.0]), AngleUnit.radian),
    (np.array([0.0, 90.0, 180.0, 270.0, 0.0]), AngleUnit.degree),
]


@pytest.mark.parametrize("phase, unit", complex_from_magnitude_phase_parameters)
@pytest.mark.validation
def test_signal_complex_from_magnitude_phase(
    phase: np.ndarray, unit: AngleUnit
) -> None:
    """Test :py:func:`sigima.proc.signal.complex_from_magnitude_phase`.

    Args:
        phase (np.ndarray): Angles in radians or degrees.
        unit (AngleUnit): Unit of the angles, either radian or degree.
    """
    x = np.linspace(0.0, 1.0, 5)
    magnitude = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    # Create signal instances for magnitude and phase
    s_mag = SignalObj("magnitude")
    s_mag.set_xydata(x, magnitude)
    s_phase = SignalObj("phase")
    s_phase.set_xydata(x, phase)
    # Create complex signal from magnitude and phase
    p = AngleUnitParam.create(unit=unit)
    result = complex_from_magnitude_phase(s_mag, s_phase, p)
    unit_str = "rad" if unit == AngleUnit.radian else "°"
    check_array_result(
        f"complex_from_magnitude_phase_{unit_str}",
        result.y,
        polar_to_complex(magnitude, phase, unit=unit_str),
    )


@pytest.mark.validation
def test_signal_astype() -> None:
    """Data type conversion validation test."""
    s1 = __create_two_signals()[0]
    for dtype_str in sigima.objects.SignalObj.get_valid_dtypenames():
        p = sigima.params.DataTypeSParam.create(dtype_str=dtype_str)
        astype_signal = sigima.proc.signal.astype(s1, p)
        assert astype_signal.y.dtype == np.dtype(dtype_str)


@pytest.mark.validation
def test_signal_exp() -> None:
    """Exponential validation test."""
    s1 = __create_two_signals()[0]
    exp_signal = sigima.proc.signal.exp(s1)
    check_array_result("Exponential", exp_signal.y, np.exp(s1.y))


@pytest.mark.validation
def test_signal_log10() -> None:
    """Logarithm base 10 validation test."""
    s1 = __create_two_signals()[0]
    log10_signal = sigima.proc.signal.log10(sigima.proc.signal.exp(s1))
    check_array_result("Logarithm base 10", log10_signal.y, np.log10(np.exp(s1.y)))


@pytest.mark.validation
def test_signal_sqrt() -> None:
    """Square root validation test."""
    s1 = sigima.tests.data.get_test_signal("paracetamol.txt")
    sqrt_signal = sigima.proc.signal.sqrt(s1)
    check_array_result("Square root", sqrt_signal.y, np.sqrt(s1.y))


@pytest.mark.validation
def test_signal_power() -> None:
    """Power validation test."""
    s1 = sigima.tests.data.get_test_signal("paracetamol.txt")
    p = sigima.params.PowerParam.create(power=2.0)
    power_signal = sigima.proc.signal.power(s1, p)
    check_array_result("Power", power_signal.y, s1.y**p.power)


@pytest.mark.validation
def test_signal_arithmetic() -> None:
    """Arithmetic operations validation test."""
    s1, s2 = __create_two_signals()
    p = sigima.params.ArithmeticParam.create()
    for operator in p.operators:
        p.operator = operator
        for factor in (0.0, 1.0, 2.0):
            p.factor = factor
            for constant in (0.0, 1.0, 2.0):
                p.constant = constant
                s3 = sigima.proc.signal.arithmetic(s1, s2, p)
                if operator == "+":
                    exp = s1.y + s2.y
                elif operator == "×":
                    exp = s1.y * s2.y
                elif operator == "-":
                    exp = s1.y - s2.y
                elif operator == "/":
                    exp = s1.y / s2.y
                exp = exp * factor + constant
                check_array_result(f"Arithmetic [{p.get_operation()}]", s3.y, exp)


if __name__ == "__main__":
    test_signal_addition()
    test_signal_average()
    test_signal_product()
    test_signal_difference()
    test_signal_quadratic_difference()
    test_signal_division()
    test_signal_addition_constant()
    test_signal_product_constant()
    test_signal_difference_constant()
    test_signal_division_constant()
    test_signal_inverse()
    test_signal_absolute()
    test_signal_real()
    test_signal_imag()
    test_signal_complex_from_real_imag()
    test_signal_phase()
    for parameters in complex_from_magnitude_phase_parameters:
        test_signal_complex_from_magnitude_phase(*parameters)
    test_signal_astype()
    test_signal_exp()
    test_signal_log10()
    test_signal_sqrt()
    test_signal_power()
    test_signal_arithmetic()
