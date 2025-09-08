# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Fit Models (see parent package :mod:`sigima.tools`)

"""

from __future__ import annotations

import abc

import numpy as np
import scipy.special


class FitModel(abc.ABC):
    """Curve fitting model base class"""

    @classmethod
    @abc.abstractmethod
    def func(cls, x, amp, sigma, x0, y0):
        """Return fitting function"""

    # pylint: disable=unused-argument
    @classmethod
    def get_amp_from_amplitude(cls, amplitude, sigma):
        """Return amp from function amplitude and sigma"""
        return amplitude

    @classmethod
    def amplitude(cls, amp, sigma):
        """Return function amplitude"""
        return cls.func(0, amp, sigma, 0, 0)

    @classmethod
    @abc.abstractmethod
    def fwhm(cls, amp, sigma):
        """Return function FWHM"""

    @classmethod
    def half_max_segment(cls, amp, sigma, x0, y0):
        """Return segment coordinates for y=half-maximum intersection"""
        hwhm = 0.5 * cls.fwhm(amp, sigma)
        yhm = 0.5 * cls.amplitude(amp, sigma) + y0
        return x0 - hwhm, yhm, x0 + hwhm, yhm


class GaussianModel(FitModel):
    """1-dimensional Gaussian fit model"""

    @classmethod
    def func(cls, x, amp, sigma, x0, y0):
        """Return fitting function"""
        return (
            amp / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - x0) / sigma) ** 2)
            + y0
        )

    @classmethod
    def get_amp_from_amplitude(cls, amplitude, sigma):
        """Return amp from function amplitude and sigma"""
        return amplitude * (sigma * np.sqrt(2 * np.pi))

    @classmethod
    def amplitude(cls, amp, sigma):
        """Return function amplitude"""
        return amp / (sigma * np.sqrt(2 * np.pi))

    @classmethod
    def fwhm(cls, amp, sigma):
        """Return function FWHM"""
        return 2 * sigma * np.sqrt(2 * np.log(2))


class LorentzianModel(FitModel):
    """1-dimensional Lorentzian fit model"""

    @classmethod
    def func(cls, x, amp, sigma, x0, y0):
        """Return fitting function"""
        return (amp / (sigma * np.pi)) / (1 + ((x - x0) / sigma) ** 2) + y0

    @classmethod
    def get_amp_from_amplitude(cls, amplitude, sigma):
        """Return amp from function amplitude and sigma"""
        return amplitude * (sigma * np.pi)

    @classmethod
    def amplitude(cls, amp, sigma):
        """Return function amplitude"""
        return amp / (sigma * np.pi)

    @classmethod
    def fwhm(cls, amp, sigma):
        """Return function FWHM"""
        return 2 * sigma


class VoigtModel(FitModel):
    """1-dimensional Voigt fit model"""

    @classmethod
    def func(cls, x, amp, sigma, x0, y0):
        """Return fitting function"""
        # pylint: disable=no-member
        z = (x - x0 + 1j * sigma) / (sigma * np.sqrt(2.0))
        return y0 + amp * scipy.special.wofz(z).real / (sigma * np.sqrt(2 * np.pi))

    @classmethod
    def fwhm(cls, amp, sigma):
        """Return function FWHM"""
        wg = GaussianModel.fwhm(amp, sigma)
        wl = LorentzianModel.fwhm(amp, sigma)
        return 0.5346 * wl + np.sqrt(0.2166 * wl**2 + wg**2)


class PlanckianModel(FitModel):
    """Planckian (blackbody radiation) fit model"""

    @classmethod
    def func(cls, x, amp, x0, sigma, y0):
        """Return Planckian fitting function

        Args:
            x: wavelength values (in nm or other units)
            amp: amplitude scaling factor
            x0: peak wavelength (Wien's displacement law)
            sigma: width parameter (larger sigma = wider peak)
            y0: baseline offset
        """
        # Planck-like function with Wien's displacement law behavior
        # The function peaks at approximately x0 when properly parameterized

        x = np.asarray(x, dtype=float)
        result = np.full_like(x, y0, dtype=float)

        # Only compute for positive wavelengths
        valid_mask = x > 0
        if not np.any(valid_mask):
            return result

        x_valid = x[valid_mask]

        try:
            # Wien's displacement law: λ_max * T = constant
            # For a proper Planckian curve, we need:
            # d/dx [x^(-5) / (exp(c/x) - 1)] = 0 at x = x0
            # This gives us c = 5*x0 for the peak condition

            # The exponential argument that produces peak at x0
            wien_constant = 5.0

            # Use sigma to control the effective temperature/width
            # sigma=1.0 gives the canonical Planck curve
            # sigma>1.0 gives broader curves (cooler)
            # sigma<1.0 gives sharper curves (hotter)
            temperature_factor = sigma

            exp_argument = wien_constant * x0 / (x_valid * temperature_factor)

            # Clip to prevent overflow
            exp_argument = np.clip(exp_argument, 0, 50)

            # Planck function components:
            # 1. The wavelength dependence: x^(-5)
            wavelength_factor = (x_valid / x0) ** (-5)

            # 2. The exponential term: 1/(exp(arg) - 1)
            exp_denominator = np.expm1(exp_argument)  # exp(x) - 1

            # Avoid division by very small numbers
            exp_denominator = np.where(
                np.abs(exp_denominator) < 1e-12, 1e-12, exp_denominator
            )

            # Combine the Planckian terms
            planck_curve = wavelength_factor / exp_denominator

            # Apply amplitude and add to baseline
            result[valid_mask] += amp * planck_curve

        except (OverflowError, ZeroDivisionError, RuntimeWarning):
            # If computation fails, return baseline only
            pass

        return result

    @classmethod
    def fwhm(cls, amp, sigma):  # pylint: disable=unused-argument
        """Return function FWHM (approximate)"""
        return 2.9 * sigma  # Empirical approximation for Planckian FWHM


class TwoHalfGaussianModel(FitModel):
    """Two half-Gaussian fit model for asymmetric peaks"""

    @classmethod
    def func(
        cls, x, amp_left, amp_right, sigma_left, sigma_right, x0, y0_left, y0_right
    ):
        """Return two half-Gaussian with separate left/right amplitudes (Matris-inspired enhancement)

        Args:
            x: x values
            amp_left: amplitude for left side (x < x0)
            amp_right: amplitude for right side (x >= x0)
            sigma_left: standard deviation for x < x0
            sigma_right: standard deviation for x > x0
            x0: center position
            y0_left: baseline offset for x < x0
            y0_right: baseline offset for x >= x0
        """
        result = np.zeros_like(x)

        # Left side (x < x0): use amp_left, sigma_left and y0_left
        left_mask = x < x0
        if np.any(left_mask):
            exp_left = np.exp(-0.5 * ((x[left_mask] - x0) / sigma_left) ** 2)
            result[left_mask] = y0_left + amp_left * exp_left

        # Right side (x >= x0): use amp_right, sigma_right and y0_right
        right_mask = x >= x0
        if np.any(right_mask):
            exp_right = np.exp(-0.5 * ((x[right_mask] - x0) / sigma_right) ** 2)
            result[right_mask] = y0_right + amp_right * exp_right

        return result

    @classmethod
    def fwhm(cls, amp, sigma):  # pylint: disable=unused-argument
        """Return function FWHM (using average of left and right sigmas)"""
        # For two half-Gaussian, we'd need both sigmas,
        # but this is the base class interface
        return 2 * sigma * np.sqrt(2 * np.log(2))


class DoubleExponentialModel(FitModel):
    """Double exponential fit model"""

    @classmethod
    def func(cls, x, x_center, a_left, b_left, a_right, b_right, y0):
        """Return double exponential fitting function

        Args:
            x: time values
            x_center: center position (boundary between left and right components)
            a_left: left component amplitude coefficient
            b_left: left component time constant coefficient
            a_right: right component amplitude coefficient
            b_right: right component time constant coefficient
            y0: baseline offset
        """
        y = np.zeros_like(x)
        y[x < x_center] = a_left * np.exp(b_left * x[x < x_center]) + y0
        y[x >= x_center] = a_right * np.exp(b_right * x[x >= x_center]) + y0
        return y

    @classmethod
    def fwhm(cls, amp, sigma):  # pylint: disable=unused-argument
        """Return function FWHM (not well-defined for double exponential)"""
        raise NotImplementedError(
            "FWHM is not well-defined for double exponential model"
        )
