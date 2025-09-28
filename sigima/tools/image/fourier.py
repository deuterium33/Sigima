# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Fourier analysis module
-----------------------

This module provides 2D Fourier transform utilities and frequency domain operations
for image processing.

Features include:

- 2D FFT/IFFT functions with optional shifting
- Spectral analysis (magnitude spectrum, phase spectrum, power spectral density)
- Frequency domain filtering and deconvolution
- Zero padding utilities for FFT operations

These tools support various frequency domain image processing operations
including filtering, spectral analysis, and deconvolution.
"""

from __future__ import annotations

import warnings

import numpy as np

from sigima.tools.checks import check_2d_array

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...


@check_2d_array
def fft2d(z: np.ndarray, shift: bool = True) -> np.ndarray:
    """Compute FFT of complex array `z`

    Args:
        z: Input data
        shift: Shift zero frequency to center (default: True)

    Returns:
        FFT of input data
    """
    z1 = np.fft.fft2(z)
    if shift:
        z1 = np.fft.fftshift(z1)
    return z1


@check_2d_array
def ifft2d(z: np.ndarray, shift: bool = True) -> np.ndarray:
    """Compute inverse FFT of complex array `z`

    Args:
        z: Input data
        shift: Shift zero frequency to center (default: True)

    Returns:
        Inverse FFT of input data
    """
    if shift:
        z = np.fft.ifftshift(z)
    z1 = np.fft.ifft2(z)
    return z1


@check_2d_array
def magnitude_spectrum(z: np.ndarray, log_scale: bool = False) -> np.ndarray:
    """Compute magnitude spectrum of complex array `z`

    Args:
        z: Input data
        log_scale: Use log scale (default: False)

    Returns:
        Magnitude spectrum of input data
    """
    z1 = np.abs(fft2d(z))
    if log_scale:
        z1 = 20 * np.log10(z1.clip(1e-10))
    return z1


@check_2d_array
def phase_spectrum(z: np.ndarray) -> np.ndarray:
    """Compute phase spectrum of complex array `z`

    Args:
        z: Input data

    Returns:
        Phase spectrum of input data (in degrees)
    """
    return np.rad2deg(np.angle(fft2d(z)))


@check_2d_array
def psd(z: np.ndarray, log_scale: bool = False) -> np.ndarray:
    """Compute power spectral density of complex array `z`

    Args:
        z: Input data
        log_scale: Use log scale (default: False)

    Returns:
        Power spectral density of input data
    """
    z1 = np.abs(fft2d(z)) ** 2
    if log_scale:
        z1 = 10 * np.log10(z1.clip(1e-10))
    return z1


@check_2d_array
def gaussian_freq_filter(
    data: np.ndarray, f0: float = 0.1, sigma: float = 0.05
) -> np.ndarray:
    """
    Apply a 2D Gaussian bandpass filter in the frequency domain to an image.

    This function performs a 2D Fast Fourier Transform (FFT) on the input image,
    applies a Gaussian filter centered at frequency `f0` with standard deviation `sigma`
    (both expressed in cycles per pixel), and then transforms the result back to the
    spatial domain.

    Args:
        data: Input image data.
        f0: Center frequency of the Gaussian filter (cycles/pixel).
        sigma: Standard deviation of the Gaussian filter (cycles/pixel).

    Returns:
        The filtered image.
    """
    n, m = data.shape
    fx = np.fft.fftshift(np.fft.fftfreq(m, d=1))
    fy = np.fft.fftshift(np.fft.fftfreq(n, d=1))
    fx_grid, fy_grid = np.meshgrid(fx, fy)
    freq_radius = np.hypot(fx_grid, fy_grid)

    # Create the 2D Gaussian bandpass filter
    gaussian_filter = np.exp(-0.5 * ((freq_radius - f0) / sigma) ** 2)

    # Apply FFT, filter in frequency domain, and inverse FFT
    fft_data = fft2d(data, shift=True)
    filtered_fft = fft_data * gaussian_filter
    zout = ifft2d(filtered_fft, shift=True)
    return zout.real


@check_2d_array(non_constant=True)
def deconvolve(
    data: np.ndarray,
    kernel: np.ndarray,
    reg: float = 0.0,
    boundary: str = "edge",
) -> np.ndarray:
    """
    Perform 2D FFT deconvolution with correct 'same' geometry (no shift).

    The kernel (PSF) must be centered (impulse at center for identity kernel).
    Odd kernel sizes are recommended.

    Args:
        data: Input image (2D array).
        kernel: Point Spread Function (PSF), centered.
        reg: Regularization parameter (if >0, Wiener/Tikhonov inverse:
         ``H* / (|H|^2 + reg))``.
        boundary: Padding mode ('edge' for constant plateau,
         'reflect' for symmetric mirror).

    Returns:
        Deconvolved image (same shape as input).

    Raises:
        ValueError: If kernel is empty or null.
    """
    if kernel.size == 0 or not np.any(kernel):
        raise ValueError("Deconvolution kernel cannot be null.")

    H, W = data.shape
    kh, kw = kernel.shape

    if kh % 2 == 0 or kw % 2 == 0:
        # Warning for even-sized kernels (off-by-one in centered FFT)
        warnings.warn(
            "Deconvolution kernel should have odd dimensions for centered FFT."
        )

    # Symmetric padding for centered 'same' convolution
    top = kh // 2
    bottom = kh - 1 - top
    left = kw // 2
    right = kw - 1 - left
    data_pad = np.pad(data, ((top, bottom), (left, right)), mode=boundary)
    Hp, Wp = data_pad.shape  # = H+kh-1, W+kw-1

    # Centered PSF to OTF conversion (avoid off-by-one for even sizes)
    kernel_pad = np.zeros_like(data_pad, dtype=float)
    r0 = Hp // 2 - kh // 2
    c0 = Wp // 2 - kw // 2
    kernel_pad[r0 : r0 + kh, c0 : c0 + kw] = kernel
    H_otf = np.fft.fft2(np.fft.ifftshift(kernel_pad))  # center → (0,0)

    # FFT of padded image (no shift)
    Z = np.fft.fft2(data_pad)

    # Frequency domain inversion
    if reg > 0.0:
        Hc = np.conj(H_otf)
        X = Z * Hc / (np.abs(H_otf) ** 2 + float(reg))
    else:
        eps = 1e-12
        X = Z / (H_otf + eps)

    data_true_pad = np.fft.ifft2(X).real

    # Central crop to restore original geometry
    out = data_true_pad[top : top + H, left : left + W]
    return out
