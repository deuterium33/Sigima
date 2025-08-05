# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Frequency filters unit tests.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

import numpy as np
import pytest

import sigima.proc.signal
from sigima.objects.signal import SignalObj, create_signal
from sigima.tests import guiutils
from sigima.tests.helpers import check_array_result, check_scalar_result
from sigima.tools.signal.fourier import brick_wall_filter

# TODO: For each test, check all filter methods (brickwall, butterworth, etc.)
#       by using something like:
#       for method_name, _method_label in sigima.proc.signal.LowPassFilterParam.methods:


def build_clean_noisy_signals(
    length: int = 2**15,
    freq: int | float | np.ndarray = 1,
    noise_level: float = 0.2,
) -> tuple[SignalObj, SignalObj]:
    """Create a test 1D signal + high-freq noise.

    Args:
        length: Length of the signal.
        freq: Frequency of the sine wave, can be a single value or an array of
         frequencies
        noise_level: Standard deviation of the Gaussian noise to be added.

    Returns:
        Tuple of (clean_signal, noisy_signal) where:
        - clean_signal: The clean sine wave signal.
        - noisy_signal: The noisy signal with added Gaussian noise.
    """
    x = np.linspace(0, 1, length)
    if np.isscalar(freq):
        y_clean = np.sin(2 * np.pi * freq * x)
    else:
        freq = np.asarray(freq)
        y_clean = np.sum([np.sin(2 * np.pi * f * x) for f in freq], axis=0)
    rng = np.random.default_rng(seed=0)
    y_noisy = y_clean + noise_level * rng.standard_normal(size=length)
    noisy = create_signal("noisy signal", x, y_noisy)
    clean = create_signal("clean signal", x, y_clean)
    return clean, noisy


@pytest.mark.validation
def test_signal_lowpass(request: pytest.FixtureRequest | None = None) -> None:
    """Validation test for frequency filtering."""
    guiutils.set_current_request(request)

    clean, noisy = build_clean_noisy_signals()

    param = sigima.proc.signal.LowPassFilterParam.create(
        method="brick_wall",
        cut0=2.0,
        zero_padding=False,
    )
    # Lowpass: should keep the sine, remove most noise
    filt = sigima.proc.signal.lowpass(noisy, param)

    if guiutils.is_gui_enabled():
        # pylint: disable=import-outside-toplevel
        from guidata.qthelpers import qt_app_context

        from sigima.tests.vistools import view_curves

        with qt_app_context():
            # Show original and filtered signals
            view_curves([clean, filt])

    # Compare filtered signal to clean signal (ignore edges)
    check_array_result(
        "brickwall lowpass noise reduction",
        filt.y[10 : len(clean.y) - 10],
        clean.y[10 : len(clean.y) - 10],
        atol=0.15,
    )


@pytest.mark.validation
def test_signal_highpass(request: pytest.FixtureRequest | None = None) -> None:
    """Validation test for highpass frequency filtering."""
    guiutils.set_current_request(request)

    noise_level = 0.2  # Set noise level for the test
    clean, noisy = build_clean_noisy_signals(noise_level=noise_level)
    param = sigima.proc.signal.HighPassFilterParam.create(
        method="brick_wall",
        cut0=2.0,
        zero_padding=False,
    )
    filt = sigima.proc.signal.highpass(noisy, param)

    if guiutils.is_gui_enabled():
        # pylint: disable=import-outside-toplevel
        from guidata.qthelpers import qt_app_context

        from sigima.tests.vistools import view_curves

        with qt_app_context():
            view_curves([clean, filt])

    # The mean of the filtered signal should be close to zero (since only noise remains)
    mean_variance = np.sqrt(noise_level / len(clean.x))
    expected_err = 3 * mean_variance
    check_scalar_result(
        "brickwall highpass removes low freq",
        float(np.mean(filt.y)),
        0,
        atol=expected_err,
    )


@pytest.mark.validation
def test_signal_stopband(request: pytest.FixtureRequest | None = None) -> None:
    """Validation test for stopband frequency filtering."""
    guiutils.set_current_request(request)

    tst_sig, _noisy = build_clean_noisy_signals(freq=np.array([1, 3, 5]), noise_level=0)
    exp_sig, _ = build_clean_noisy_signals(freq=np.array([1, 5]), noise_level=0)

    param = sigima.proc.signal.BandStopFilterParam.create(
        method="brick_wall",
        cut0=2.0,
        cut1=4.0,
        zero_padding=False,
    )
    res_sig = sigima.proc.signal.bandstop(tst_sig, param)

    if guiutils.is_gui_enabled():
        # pylint: disable=import-outside-toplevel
        from guidata.qthelpers import qt_app_context

        from sigima.tests.vistools import view_curves

        with qt_app_context():
            view_curves([exp_sig, res_sig])

    check_array_result(
        "brickwall stopband",
        res_sig.y[10 : len(res_sig.y) - 10],
        exp_sig.y[10 : len(exp_sig.y) - 10],
        atol=1e-3,
    )


@pytest.mark.validation
def test_signal_bandpass(request: pytest.FixtureRequest | None = None) -> None:
    """Validation test for bandpass frequency filtering."""
    guiutils.set_current_request(request)

    tst_sig, _noisy = build_clean_noisy_signals(freq=np.array([1, 3, 5]), noise_level=0)
    exp_sig, _ = build_clean_noisy_signals(freq=np.array([3]), noise_level=0)
    param = sigima.proc.signal.BandPassFilterParam.create(
        method="brick_wall",
        cut0=2.0,
        cut1=4.0,
        zero_padding=False,
    )
    res_sig = sigima.proc.signal.bandpass(tst_sig, param)

    if guiutils.is_gui_enabled():
        # pylint: disable=import-outside-toplevel
        from guidata.qthelpers import qt_app_context

        from sigima.tests.vistools import view_curves

        with qt_app_context():
            view_curves([exp_sig, res_sig])

    check_array_result(
        "brickwall bandpass",
        res_sig.y[10 : len(exp_sig.y) - 10],
        exp_sig.y[10 : len(exp_sig.y) - 10],
        atol=1e-3,
    )


def test_brickwall_filter_invalid_x():
    """Test brickwall_filter raises on non-uniform x."""
    clean, noisy = build_clean_noisy_signals()
    x_bad = clean.x.copy()
    x_bad[5] += 0.01  # break uniformity
    with pytest.raises(ValueError, match="evenly spaced"):
        brick_wall_filter(x_bad, noisy.y, "lowpass", cutoff0=0.1)


def test_tools_to_proc_interface():
    """Test that the `brickwall_filter` function is properly interfaced
    with the `sigima.proc` module, via the `lowpass`, `highpass`, `bandpass`,
    and `stopband` functions.
    """
    _clean, tst_sig = build_clean_noisy_signals(freq=np.array([1, 3, 5]))

    # Lowpass
    tools_res = brick_wall_filter(tst_sig.x, tst_sig.y, "lowpass", cutoff0=2.0)
    proc_res = sigima.proc.signal.lowpass(
        tst_sig,
        sigima.proc.signal.LowPassFilterParam.create(
            cut0=2.0, method="brick_wall", zero_padding=False
        ),
    )
    check_array_result("Lowpass filter result", tools_res[1], proc_res.y, atol=1e-3)

    # Highpass
    tools_res = brick_wall_filter(tst_sig.x, tst_sig.y, "highpass", cutoff0=2.0)
    proc_res = sigima.proc.signal.highpass(
        tst_sig,
        sigima.proc.signal.HighPassFilterParam.create(
            cut0=2.0, method="brick_wall", zero_padding=False
        ),
    )
    check_array_result("Highpass filter result", tools_res[1], proc_res.y, atol=1e-3)

    # Bandpass
    tools_res = brick_wall_filter(
        tst_sig.x, tst_sig.y, "bandpass", cutoff0=2.0, cutoff1=4.0
    )
    proc_res = sigima.proc.signal.bandpass(
        tst_sig,
        sigima.proc.signal.BandPassFilterParam.create(
            cut0=2.0, cut1=4.0, method="brick_wall", zero_padding=False
        ),
    )
    check_array_result("Bandpass filter result", tools_res[1], proc_res.y, atol=1e-3)

    # Bandstop
    tools_res = brick_wall_filter(
        tst_sig.x, tst_sig.y, "bandstop", cutoff0=2.0, cutoff1=4.0
    )
    proc_res = sigima.proc.signal.bandstop(
        tst_sig,
        sigima.proc.signal.BandStopFilterParam.create(
            cut0=2.0, cut1=4.0, method="brick_wall", zero_padding=False
        ),
    )
    check_array_result("Bandstop filter result", tools_res[1], proc_res.y, atol=1e-3)


if __name__ == "__main__":
    test_signal_lowpass(request=guiutils.DummyRequest(gui=True))
    test_signal_highpass(request=guiutils.DummyRequest(gui=True))
    test_signal_stopband(request=guiutils.DummyRequest(gui=True))
    test_signal_bandpass(request=guiutils.DummyRequest(gui=True))
    test_brickwall_filter_invalid_x()
    test_tools_to_proc_interface()
