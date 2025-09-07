# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for exposure computation functions.
"""

from __future__ import annotations

import numpy as np
import pytest
from skimage import exposure

import sigima.enums
import sigima.objects
import sigima.params
import sigima.proc.image
from sigima.tests.data import get_test_image
from sigima.tests.helpers import check_array_result, check_scalar_result


@pytest.mark.validation
def test_adjust_gamma() -> None:
    """Validation test for the image gamma adjustment processing."""
    # See [1] for more information about the validation of exposure methods.
    src = get_test_image("flower.npy")
    for gamma, gain in ((0.5, 1.0), (1.0, 2.0), (1.5, 0.5)):
        p = sigima.params.AdjustGammaParam.create(gamma=gamma, gain=gain)
        dst = sigima.proc.image.adjust_gamma(src, p)
        exp = exposure.adjust_gamma(src.data, gamma=gamma, gain=gain)
        check_array_result(f"AdjustGamma[gamma={gamma},gain={gain}]", dst.data, exp)


@pytest.mark.validation
def test_adjust_log() -> None:
    """Validation test for the image logarithmic adjustment processing."""
    # See [1] for more information about the validation of exposure methods.
    src = get_test_image("flower.npy")
    for gain, inv in ((1.0, False), (2.0, True)):
        p = sigima.params.AdjustLogParam.create(gain=gain, inv=inv)
        dst = sigima.proc.image.adjust_log(src, p)
        exp = exposure.adjust_log(src.data, gain=gain, inv=inv)
        check_array_result(f"AdjustLog[gain={gain},inv={inv}]", dst.data, exp)


@pytest.mark.validation
def test_adjust_sigmoid() -> None:
    """Validation test for the image sigmoid adjustment processing."""
    # See [1] for more information about the validation of exposure methods.
    src = get_test_image("flower.npy")
    for cutoff, gain, inv in ((0.5, 1.0, False), (0.25, 2.0, True)):
        p = sigima.params.AdjustSigmoidParam.create(cutoff=cutoff, gain=gain, inv=inv)
        dst = sigima.proc.image.adjust_sigmoid(src, p)
        exp = exposure.adjust_sigmoid(src.data, cutoff=cutoff, gain=gain, inv=inv)
        check_array_result(
            f"AdjustSigmoid[cutoff={cutoff},gain={gain},inv={inv}]", dst.data, exp
        )


@pytest.mark.validation
def test_rescale_intensity() -> None:
    """Validation test for the image intensity rescaling processing."""
    # See [1] for more information about the validation of exposure methods.
    src = get_test_image("flower.npy")
    p = sigima.params.RescaleIntensityParam.create(in_range="dtype", out_range="image")
    dst = sigima.proc.image.rescale_intensity(src, p)
    exp = exposure.rescale_intensity(
        src.data, in_range=p.in_range, out_range=p.out_range
    )
    check_array_result("RescaleIntensity", dst.data, exp)


@pytest.mark.validation
def test_equalize_hist() -> None:
    """Validation test for the image histogram equalization processing."""
    # See [1] for more information about the validation of exposure methods.
    src = get_test_image("flower.npy")
    for nbins in (256, 512):
        p = sigima.params.EqualizeHistParam.create(nbins=nbins)
        dst = sigima.proc.image.equalize_hist(src, p)
        exp = exposure.equalize_hist(src.data, nbins=nbins)
        check_array_result(f"EqualizeHist[nbins={nbins}]", dst.data, exp)


@pytest.mark.validation
def test_equalize_adapthist() -> None:
    """Validation test for the image adaptive histogram equalization processing."""
    # See [1] for more information about the validation of exposure methods.
    src = get_test_image("flower.npy")
    for clip_limit in (0.01, 0.1):
        p = sigima.params.EqualizeAdaptHistParam.create(clip_limit=clip_limit)
        dst = sigima.proc.image.equalize_adapthist(src, p)
        exp = exposure.equalize_adapthist(src.data, clip_limit=clip_limit)
        check_array_result(f"AdaptiveHist[clip_limit={clip_limit}]", dst.data, exp)


@pytest.mark.validation
def test_image_normalize() -> None:
    """Validation test for the image normalization processing."""
    src = get_test_image("flower.npy")
    src.data = np.array(src.data, dtype=float)
    src.data[20:30, 20:30] = np.nan  # Adding NaN values to the image
    p = sigima.params.NormalizeParam()

    # Given the fact that the normalization methods implementations are
    # straightforward, we do not need to compare arrays with each other,
    # we simply need to check if some properties are satisfied.
    for method in sigima.enums.NormalizationMethod:
        p.method = method
        dst = sigima.proc.image.normalize(src, p)
        title = f"Normalize[method='{p.method}']"
        exp_min, exp_max = None, None
        if p.method == sigima.enums.NormalizationMethod.MAXIMUM:
            exp_min, exp_max = np.nanmin(src.data) / np.nanmax(src.data), 1.0
        elif p.method == sigima.enums.NormalizationMethod.AMPLITUDE:
            exp_min, exp_max = 0.0, 1.0
        elif p.method == sigima.enums.NormalizationMethod.AREA:
            area = np.nansum(src.data)
            exp_min, exp_max = np.nanmin(src.data) / area, np.nanmax(src.data) / area
        elif p.method == sigima.enums.NormalizationMethod.ENERGY:
            energy = np.sqrt(np.nansum(np.abs(src.data) ** 2))
            exp_min, exp_max = (
                np.nanmin(src.data) / energy,
                np.nanmax(src.data) / energy,
            )
        elif p.method == sigima.enums.NormalizationMethod.RMS:
            rms = np.sqrt(np.nanmean(np.abs(src.data) ** 2))
            exp_min, exp_max = np.nanmin(src.data) / rms, np.nanmax(src.data) / rms
        check_scalar_result(f"{title}|min", np.nanmin(dst.data), exp_min)
        check_scalar_result(f"{title}|max", np.nanmax(dst.data), exp_max)


@pytest.mark.validation
def test_image_calibration() -> None:
    """Validation test for the image calibration processing."""
    src = get_test_image("flower.npy")
    p = sigima.params.XYZCalibrateParam()
    for axis in ("x", "y", "z"):
        for a, b in ((1.0, 0.0), (0.5, 0.1)):
            p.axis = axis
            p.a, p.b = a, b
            dst = sigima.proc.image.calibration(src, p)
            exp = src.copy("expected")
            if p.a == 1.0 and p.b == 0.0:
                suffix = "identity"
                # Identity, do nothing except convert data to float for Z-axis case
                if axis == "z":
                    exp.data = np.array(src.data, dtype=float)
            else:
                suffix = f"a={p.a},b={p.b}"
                if axis in ("x", "y"):
                    setattr(exp, f"{axis}0", getattr(src, f"{axis}0") * p.a + p.b)
                    setattr(exp, f"d{axis}", getattr(src, f"d{axis}") * p.a)
                else:
                    exp.data = p.a * src.data + p.b
            title = f"Calibration[{axis},{suffix}]"
            if axis in ("x", "y"):
                x0n, dxn = f"{axis}0", f"d{axis}"
                res_x0n, exp_x0n = getattr(dst, x0n), getattr(exp, x0n)
                res_dxn, exp_dxn = getattr(dst, dxn), getattr(exp, dxn)
                check_scalar_result(f"{title}|{x0n}", res_x0n, exp_x0n)
                check_scalar_result(f"{title}|{dxn}", res_dxn, exp_dxn)
            check_array_result(title, dst.data, exp.data)


@pytest.mark.validation
def test_image_clip() -> None:
    """Validation test for the image clipping processing."""
    src = get_test_image("flower.npy")
    p = sigima.params.ClipParam()

    for lower, upper in ((float("-inf"), float("inf")), (50, 100)):
        p.lower, p.upper = lower, upper
        dst = sigima.proc.image.clip(src, p)
        exp = np.clip(src.data, p.lower, p.upper)
        check_array_result(f"Clip[{lower},{upper}]", dst.data, exp)


@pytest.mark.validation
def test_image_offset_correction() -> None:
    """Validation test for the image offset correction processing."""
    src = get_test_image("flower.npy")
    # Defining the ROI that will be used to estimate the offset
    p = sigima.objects.ROI2DParam.create(x0=0, y0=0, dx=50, dy=20)
    dst = sigima.proc.image.offset_correction(src, p)
    ix0, iy0 = int(p.x0), int(p.y0)
    ix1, iy1 = int(p.x0 + p.dx), int(p.y0 + p.dy)
    exp = src.data - np.mean(src.data[iy0:iy1, ix0:ix1])
    check_array_result("OffsetCorrection", dst.data, exp)


if __name__ == "__main__":
    test_adjust_gamma()
    test_adjust_log()
    test_adjust_sigmoid()
    test_rescale_intensity()
    test_equalize_hist()
    test_equalize_adapthist()
    test_image_normalize()
    test_image_calibration()
    test_image_clip()
    test_image_offset_correction()
