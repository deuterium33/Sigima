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
    def fwhm(cls, amp, sigma):
        """Return function FWHM (approximate)"""
        return 2.9 * sigma  # Empirical approximation for Planckian FWHM


class TwoHalfGaussianModel(FitModel):
    """Two half-Gaussian fit model for asymmetric peaks"""

    @classmethod
    def func(cls, x, amp, sigma_left, sigma_right, x0, y0):
        """Return two half-Gaussian fitting function

        Args:
            x: x values
            amp: amplitude (peak height above baseline)
            sigma_left: standard deviation for x < x0
            sigma_right: standard deviation for x > x0
            x0: center position
            y0: baseline offset
        """
        result = np.zeros_like(x) + y0

        # Ensure continuity at x0 by using the same amplitude for both sides
        # The amplitude represents the peak height above baseline

        # Left side (x < x0): use sigma_left
        left_mask = x < x0
        if np.any(left_mask):
            exp_left = np.exp(-0.5 * ((x[left_mask] - x0) / sigma_left) ** 2)
            result[left_mask] += amp * exp_left

        # Right side (x >= x0): use sigma_right
        right_mask = x >= x0
        if np.any(right_mask):
            exp_right = np.exp(-0.5 * ((x[right_mask] - x0) / sigma_right) ** 2)
            result[right_mask] += amp * exp_right

        return result

    @classmethod
    def fwhm(cls, amp, sigma):
        """Return function FWHM (using average of left and right sigmas)"""
        # For two half-Gaussian, we'd need both sigmas,
        # but this is the base class interface
        return 2 * sigma * np.sqrt(2 * np.log(2))


class DoubleExponentialModel(FitModel):
    """Double exponential decay fit model"""

    @classmethod
    def func(cls, x, amp1, amp2, tau1, tau2, y0):
        """Return double exponential fitting function

        Args:
            x: time values
            amp1: amplitude of first exponential
            amp2: amplitude of second exponential
            tau1: time constant of first exponential
            tau2: time constant of second exponential
            y0: baseline offset
        """
        # Prevent numerical issues with very small time constants
        tau1 = np.maximum(tau1, 1e-12)
        tau2 = np.maximum(tau2, 1e-12)

        # Auto-order time constants: ensure tau1 < tau2 (fast < slow)
        # This provides consistent parameter interpretation
        if tau1 > tau2:
            tau1, tau2 = tau2, tau1
            amp1, amp2 = amp2, amp1

        # Ensure x is positive for physical interpretation
        x = np.maximum(x, 0)

        # Calculate exponentials with overflow protection
        exp1 = np.exp(-np.minimum(x / tau1, 50))  # Prevent overflow
        exp2 = np.exp(-np.minimum(x / tau2, 50))  # Prevent overflow

        return amp1 * exp1 + amp2 * exp2 + y0

    @classmethod
    def fwhm(cls, amp, sigma):
        """Return function FWHM (not well-defined for double exponential)"""
        # For double exponential, FWHM is not well-defined, return tau equivalent
        return sigma * np.log(2)
        # For double exponential, FWHM is not well-defined, return tau equivalent
        return sigma * np.log(2)
