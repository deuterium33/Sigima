# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for restoration computation functions.
"""

from __future__ import annotations

import pytest
from skimage import morphology, restoration

import sigima.objects
import sigima.params
import sigima.proc.image
from sigima.tests.data import get_test_image
from sigima.tests.helpers import check_array_result


@pytest.mark.validation
def test_denoise_tv() -> None:
    """Validation test for the image Total Variation denoising processing."""
    # See [1] for more information about the validation of restoration methods.
    src = get_test_image("flower.npy")
    src.data = src.data[::8, ::8]
    for weight, eps, mni in ((0.1, 0.0002, 200), (0.5, 0.0001, 100)):
        p = sigima.params.DenoiseTVParam.create(
            weight=weight, eps=eps, max_num_iter=mni
        )
        dst = sigima.proc.image.denoise_tv(src, p)
        exp = restoration.denoise_tv_chambolle(src.data, weight, eps, mni)
        check_array_result(
            f"DenoiseTV[weight={weight},eps={eps},max_num_iter={mni}]",
            dst.data,
            exp,
        )


@pytest.mark.validation
def test_denoise_bilateral() -> None:
    """Validation test for the image bilateral denoising processing."""
    # See [1] for more information about the validation of restoration methods.
    src = get_test_image("flower.npy")
    src.data = src.data[::8, ::8]
    for sigma, mode in ((1.0, "constant"), (2.0, "edge")):
        p = sigima.params.DenoiseBilateralParam.create(sigma_spatial=sigma, mode=mode)
        dst = sigima.proc.image.denoise_bilateral(src, p)
        exp = restoration.denoise_bilateral(src.data, sigma_spatial=sigma, mode=mode)
        check_array_result(
            f"DenoiseBilateral[sigma_spatial={sigma},mode={mode}]",
            dst.data,
            exp,
        )


@pytest.mark.validation
def test_denoise_wavelet() -> None:
    """Validation test for the image wavelet denoising processing."""
    # See [1] for more information about the validation of restoration methods.
    src = get_test_image("flower.npy")
    src.data = src.data[::8, ::8]
    p = sigima.params.DenoiseWaveletParam()
    for wavelets in ("db1", "db2", "db3"):
        for mode in p.modes:
            for method in ("BayesShrink",):
                p.wavelets, p.mode, p.method = wavelets, mode, method
                dst = sigima.proc.image.denoise_wavelet(src, p)
                exp = restoration.denoise_wavelet(
                    src.data, wavelet=wavelets, mode=mode, method=method
                )
                check_array_result(
                    f"DenoiseWavelet[wavelets={wavelets},mode={mode},method={method}]",
                    dst.data,
                    exp,
                    atol=0.1,
                )


@pytest.mark.validation
def test_denoise_tophat() -> None:
    """Validation test for the image top-hat denoising processing."""
    # See [1] for more information about the validation of restoration methods.
    src = get_test_image("flower.npy")
    p = sigima.params.MorphologyParam.create(radius=10)
    dst = sigima.proc.image.denoise_tophat(src, p)
    footprint = morphology.disk(p.radius)
    exp = src.data - morphology.white_tophat(src.data, footprint=footprint)
    check_array_result(f"DenoiseTophat[radius={p.radius}]", dst.data, exp)


if __name__ == "__main__":
    test_denoise_tv()
    test_denoise_bilateral()
    test_denoise_wavelet()
    test_denoise_tophat()
