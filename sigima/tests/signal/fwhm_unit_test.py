# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for full width computing features
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

import pytest

import sigima.objects
import sigima.params
import sigima.proc.signal
import sigima.tests.data
import sigima.tests.helpers
from sigima.tests.env import execenv


def __test_fwhm_interactive(obj: sigima.objects.SignalObj, method: str) -> None:
    """Interactive test for the full width at half maximum computation."""
    # pylint: disable=import-outside-toplevel
    from plotpy.builder import make

    from sigima.tests.vistools import view_curve_items

    param = sigima.params.FWHMParam.create(method=method)
    geometry = sigima.proc.signal.fwhm(obj, param)
    x0, y0, x1, y1 = geometry.coords[0]
    x, y = obj.xydata
    view_curve_items(
        [
            make.mcurve(x.real, y.real, label=obj.title),
            make.annotated_segment(x0, y0, x1, y1),
        ],
        title=f"FWHM [{method}]",
    )


@pytest.mark.gui
def test_signal_fwhm_interactive() -> None:
    """FWHM interactive test."""
    # pylint: disable=import-outside-toplevel
    from guidata.qthelpers import qt_app_context

    with qt_app_context():
        execenv.print("Computing FWHM of a multi-peak signal:")
        obj1 = sigima.tests.data.create_paracetamol_signal()
        p = sigima.tests.data.GaussianNoiseParam.create(sigma=0.05)
        obj2 = sigima.tests.data.create_noisy_signal(p)
        for method, _mname in sigima.params.FWHMParam.methods:
            execenv.print(f"  Method: {method}")
            for obj in (obj1, obj2):
                if method == "zero-crossing":
                    # Check that a warning is raised when using the zero-crossing method
                    with pytest.warns(UserWarning):
                        __test_fwhm_interactive(obj, method)
                else:
                    __test_fwhm_interactive(obj, method)


@pytest.mark.validation
def test_signal_fwhm() -> None:
    """Validation test for the full width at half maximum computation."""
    obj = sigima.tests.data.get_test_signal("fwhm.txt")
    real_fwhm = 2.675  # Manual validation
    for method, exp in (
        ("gauss", 2.40323),
        ("lorentz", 2.78072),
        ("voigt", 2.56591),
        ("zero-crossing", real_fwhm),
    ):
        param = sigima.params.FWHMParam.create(method=method)
        geometry = sigima.proc.signal.fwhm(obj, param)
        length = geometry.segments_lengths()[0]
        sigima.tests.helpers.check_scalar_result(
            f"FWHM[{method}]", length, exp, rtol=0.05
        )
    obj = sigima.tests.data.create_paracetamol_signal()
    with pytest.warns(UserWarning):
        sigima.proc.signal.fwhm(
            obj, sigima.params.FWHMParam.create(method="zero-crossing")
        )


@pytest.mark.validation
def test_signal_fw1e2() -> None:
    """Validation test for the full width at 1/e^2 maximum computation."""
    obj = sigima.tests.data.get_test_signal("fw1e2.txt")
    exp = 4.06  # Manual validation
    geometry = sigima.proc.signal.fw1e2(obj)
    length = geometry.segments_lengths()[0]
    sigima.tests.helpers.check_scalar_result("FW1E2", length, exp, rtol=0.005)


@pytest.mark.validation
def test_signal_full_width_at_y() -> None:
    """Validation test for the full width at y computation."""
    obj = sigima.tests.data.get_test_signal("fwhm.txt")
    real_fwhm = 2.675  # Manual validation
    param = sigima.params.OrdinateParam.create(y=0.5)
    geometry = sigima.proc.signal.full_width_at_y(obj, param)
    length = geometry.segments_lengths()[0]
    sigima.tests.helpers.check_scalar_result("∆X", length, real_fwhm, rtol=0.05)


if __name__ == "__main__":
    test_signal_fwhm_interactive()
    test_signal_fwhm()
    test_signal_fw1e2()
    test_signal_full_width_at_y()
