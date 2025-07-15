# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Signal FFT unit test."""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

import numpy as np
import pytest
import scipy.signal as sps

import sigima.objects
import sigima.params
import sigima.proc.signal as sigima_signal
import sigima.tests.data as ctd
from sigima.tests import guiutils
from sigima.tests.data import get_test_signal
from sigima.tests.env import execenv
from sigima.tests.helpers import check_array_result, check_scalar_result
from sigima.tools.signal import fourier


@pytest.mark.validation
def test_signal_zero_padding() -> None:
    """1D FFT zero padding validation test."""
    s1 = ctd.create_periodic_signal(
        sigima.objects.SignalTypes.COSINUS, freq=50.0, size=1000
    )

    # Validate padding length computation
    for strategy, expected_length in (
        ("next_pow2", 24),
        ("double", 1000),
        ("triple", 2000),
    ):
        param = sigima.params.ZeroPadding1DParam.create(strategy=strategy)
        param.update_from_obj(s1)
        assert param.n == expected_length, (
            f"Wrong length for '{param.strategy}' strategy: {param.n}"
            f" (expected {expected_length})"
        )

    # Validate zero padding
    param = sigima.params.ZeroPadding1DParam.create(strategy="custom", n=250)
    assert param.n is not None
    for location in ("append", "prepend", "both"):
        execenv.print(f"Validating zero padding with location = {location}...")
        param.location = location
        param.update_from_obj(s1)
        s2 = sigima_signal.zero_padding(s1, param)
        len1 = s1.y.size
        n = param.n
        exp_len2 = len1 + n
        assert s2.y.size == exp_len2, f"Wrong length: {len(s2.y)} (expected {exp_len2})"
        if location == "append":
            dx = s1.x[1] - s1.x[0]
            expected_x = np.pad(
                s1.x,
                (0, n),
                mode="linear_ramp",
                end_values=(s1.x[-1] + dx * n,),
            )
            check_array_result(f"{location}: Check x-data", s2.x, expected_x)
            check_array_result(f"{location}: Check original y-data", s2.y[:len1], s1.y)
            check_array_result(
                f"{location}: Check padded y-data", s2.y[len1:], np.zeros(n)
            )
        elif location == "prepend":
            dx = s1.x[1] - s1.x[0]
            expected_x = np.pad(
                s1.x,
                (n, 0),
                mode="linear_ramp",
                end_values=(s1.x[0] - dx * n,),
            )
            check_array_result(f"{location}: Check x-data", s2.x, expected_x)
            check_array_result(f"{location}: Check original y-data", s2.y[-len1:], s1.y)
            check_array_result(
                f"{location}: Check padded y-data", s2.y[:n], np.zeros(n)
            )
        elif location == "both":
            dx = s1.x[1] - s1.x[0]
            expected_x = np.pad(
                s1.x,
                (n // 2, n - n // 2),
                mode="linear_ramp",
                end_values=(
                    s1.x[0] - dx * (n // 2),
                    s1.x[-1] + dx * (n - n // 2),
                ),
            )
            check_array_result(f"{location}: Check x-data", s2.x, expected_x)
            check_array_result(
                f"{location}: Check original y-data", s2.y[n // 2 : n // 2 + len1], s1.y
            )
            check_array_result(
                f"{location}: Check padded y-data (before)",
                s2.y[: n // 2],
                np.zeros(n // 2),
            )
            check_array_result(
                f"{location}: Check padded y-data (after)",
                s2.y[-(n - n // 2) :],
                np.zeros(n - n // 2),
            )
        execenv.print("OK")


@pytest.mark.validation
def test_signal_fft() -> None:
    """1D FFT validation test."""
    freq = 50.0
    size = 10000

    # See note in the interactive test above
    xmin = 0.0

    s1 = ctd.create_periodic_signal(
        sigima.objects.SignalTypes.COSINUS, freq=freq, size=size, xmin=xmin
    )
    fft = sigima_signal.fft(s1)
    ifft = sigima_signal.ifft(fft)

    # Check that the inverse FFT reconstructs the original signal
    check_array_result("Cosine signal FFT/iFFT X reconstruction", s1.y, ifft.y.real)
    check_array_result("Cosine signal FFT/iFFT Y reconstruction", s1.x, ifft.x.real)

    # Check FFT properties
    mag = np.abs(fft.y)

    # Find the peak in the FFT
    ipk1, ipk2 = np.argmax(mag[: size // 2]), np.argmax(mag[size // 2 :]) + size // 2
    fpk1, fpk2 = fft.x[ipk1], fft.x[ipk2]

    # Verify the peak frequencies are correct
    check_scalar_result("Cosine negative frequency", fpk1, -freq, rtol=0.001)
    check_scalar_result("Cosine positive frequency", fpk2, freq, rtol=0.001)

    # Verify the magnitude at the peak
    exp_mag = size / 2
    check_scalar_result("Cosine peak magnitude", mag[ipk1], exp_mag, rtol=0.05)

    # Verify the symmetry of the FFT
    check_array_result(
        "FFT symmetry",
        mag[1 : size // 2],
        mag[1 + size // 2 :][::-1],
    )


@pytest.mark.validation
def test_signal_ifft(request: pytest.FixtureRequest | None = None) -> None:
    """1D iFFT validation test."""
    # To validate the iFFT, we check that the original signal and the
    # reconstructed signal are equal. We need to set the request
    # to enable the GUI if needed, so that the test can be run interactively.

    guiutils.set_current_request(request)

    newparam = sigima.objects.NewSignalParam.create(
        stype=sigima.objects.SignalTypes.COSINUS, size=500
    )

    # *** Note ***
    #
    # We set xmin to 0.0 to be able to compare the X data of the original and
    # reconstructed signals, because the FFT do not preserve the X data (phase is
    # lost, sampling rate is assumed to be constant), so that comparing the X data
    # is not meaningful if xmin is different.
    newparam.xmin = 0.0

    extra_param = sigima.objects.PeriodicParam()
    s1 = sigima.objects.create_signal_from_param(newparam, extra_param=extra_param)
    t1, y1 = s1.xydata
    for shift in (True, False):
        f1, sp1 = fourier.fft1d(t1, y1, shift=shift)
        t2, y2 = fourier.ifft1d(f1, sp1)

        execenv.print(
            f"Comparing original and recovered signals for `shift={shift}`...",
            end=" ",
        )
        check_array_result("Original and recovered x data", t2, t1, verbose=False)
        check_array_result("Original and recovered y data", y2, y1, verbose=False)
        execenv.print("OK")

        if guiutils.is_gui_enabled():
            # pylint: disable=import-outside-toplevel
            from guidata.qthelpers import qt_app_context

            from sigima.tests.vistools import view_curves

            with qt_app_context():
                view_curves(
                    [
                        s1,
                        sigima.objects.create_signal("Recovered", t2, y2),
                        sigima.objects.create_signal("Difference", t1, np.abs(y2 - y1)),
                    ]
                )


@pytest.mark.validation
def test_signal_magnitude_spectrum(
    request: pytest.FixtureRequest | None = None,
) -> None:
    """1D magnitude spectrum validation test."""
    guiutils.set_current_request(request)

    freq = 50.0
    size = 10000

    s1 = ctd.create_periodic_signal(
        sigima.objects.SignalTypes.COSINUS, freq=freq, size=size
    )
    fft = sigima_signal.fft(s1)
    mag = sigima_signal.magnitude_spectrum(s1)
    fpk1 = fft.x[np.argmax(mag.y[: size // 2])]
    check_scalar_result("Cosine negative frequency", fpk1, -freq, rtol=0.001)

    # Check that the magnitude spectrum is correct
    check_array_result("Cosine signal magnitude spectrum X", mag.x, fft.x.real)
    check_array_result("Cosine signal magnitude spectrum Y", mag.y, np.abs(fft.y))

    if guiutils.is_gui_enabled():
        # pylint: disable=import-outside-toplevel
        from guidata.qthelpers import qt_app_context

        from sigima.tests.vistools import view_curves

        with qt_app_context():
            view_curves(
                [
                    sigima.objects.create_signal("FFT-real", fft.x.real, fft.x.real),
                    sigima.objects.create_signal("FFT-imag", fft.x.real, fft.y.imag),
                    sigima.objects.create_signal("FFT-magnitude", mag.x.real, mag.y),
                ]
            )


@pytest.mark.validation
def test_signal_phase_spectrum(request: pytest.FixtureRequest | None = None) -> None:
    """1D phase spectrum validation test."""
    guiutils.set_current_request(request)

    freq = 50.0
    size = 10000

    s1 = ctd.create_periodic_signal(
        sigima.objects.SignalTypes.COSINUS, freq=freq, size=size
    )
    fft = sigima_signal.fft(s1)
    phase = sigima_signal.phase_spectrum(s1)
    fpk1 = fft.x[np.argmax(phase.y[: size // 2])]
    check_scalar_result("Cosine negative frequency", fpk1, -freq, rtol=0.001)

    # Check that the phase spectrum is correct
    check_array_result("Cosine signal phase spectrum X", phase.x, fft.x.real)
    exp_phase = np.rad2deg(np.angle(fft.y))
    check_array_result("Cosine signal phase spectrum Y", phase.y, exp_phase)

    if guiutils.is_gui_enabled():
        # pylint: disable=import-outside-toplevel
        from guidata.qthelpers import qt_app_context

        from sigima.tests.vistools import view_curves

        with qt_app_context():
            view_curves(
                [
                    sigima.objects.create_signal("FFT-real", fft.x.real, fft.x.real),
                    sigima.objects.create_signal("FFT-imag", fft.x.real, fft.y.imag),
                    sigima.objects.create_signal("Phase", phase.x.real, phase.y),
                ]
            )


@pytest.mark.validation
def test_signal_psd(request: pytest.FixtureRequest | None = None) -> None:
    """1D Power Spectral Density validation test."""
    guiutils.set_current_request(request)

    freq = 50.0
    size = 10000

    s1 = ctd.create_periodic_signal(
        sigima.objects.SignalTypes.COSINUS, freq=freq, size=size
    )
    param = sigima.params.SpectrumParam()
    for decibel in (False, True):
        param.decibel = decibel
        psd = sigima_signal.psd(s1, param)

        # Check that the PSD is correct (Welch's method is used by default)
        exp_x, exp_y = sps.welch(s1.y, fs=1.0 / (s1.x[1] - s1.x[0]))

        if decibel:
            exp_y = 10 * np.log10(exp_y)

        check_array_result(f"Cosine signal PSD X (db={decibel})", psd.x, exp_x)
        check_array_result(f"Cosine signal PSD Y (db={decibel})", psd.y, exp_y)

        if guiutils.is_gui_enabled():
            # pylint: disable=import-outside-toplevel
            from guidata.qthelpers import qt_app_context

            from sigima.tests.vistools import view_curves

            with qt_app_context():
                view_curves(
                    [
                        sigima.objects.create_signal("PSD", psd.x, psd.y),
                    ]
                )


def test_signal_spectrum(request: pytest.FixtureRequest | None = None) -> None:
    """Test several FFT-related functions on `dynamic_parameters.txt`."""
    guiutils.set_current_request(request)

    # pylint: disable=import-outside-toplevel
    from guidata.qthelpers import qt_app_context

    from sigima.tests.vistools import view_curves

    with qt_app_context():
        sig = get_test_signal("dynamic_parameters.txt")
        view_curves([sig])
        p = sigima.params.SpectrumParam.create(decibel=True)
        ms = sigima_signal.magnitude_spectrum(sig, p)
        view_curves([ms], title="Magnitude spectrum")
        ps = sigima_signal.phase_spectrum(sig)
        view_curves([ps], title="Phase spectrum")
        psd = sigima_signal.psd(sig, p)
        view_curves([psd], title="Power spectral density")


if __name__ == "__main__":
    test_signal_zero_padding()
    test_signal_fft()
    test_signal_ifft(request=guiutils.DummyRequest(gui=True))
    test_signal_magnitude_spectrum(request=guiutils.DummyRequest(gui=True))
    test_signal_phase_spectrum(request=guiutils.DummyRequest(gui=True))
    test_signal_psd(request=guiutils.DummyRequest(gui=True))
    test_signal_spectrum(request=guiutils.DummyRequest(gui=True))
