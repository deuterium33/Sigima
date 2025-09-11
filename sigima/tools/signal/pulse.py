# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Pulse analysis (see parent package :mod:`sigima.tools.signal`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import abc
import warnings
from typing import Literal

import numpy as np
import scipy.optimize
import scipy.special

from sigima.tools.checks import check_1d_arrays
from sigima.tools.signal import features, peakdetection


class PulseFitModel(abc.ABC):
    """Base class for 1D pulse fit models"""

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


class GaussianModel(PulseFitModel):
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


class LorentzianModel(PulseFitModel):
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


class VoigtModel(PulseFitModel):
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


# MARK: Pulse analysis -----------------------------------------------------------------


@check_1d_arrays(x_sorted=True)
def full_width_at_y(
    x: np.ndarray, y: np.ndarray, level: float
) -> tuple[float, float, float, float]:
    """Compute the full width at a given y level using zero-crossing method.

    Args:
        x: X data
        y: Y data
        level: The Y level at which to compute the width

    Returns:
        Full width segment coordinates
    """
    crossings = features.find_all_x_at_given_y_value(x, y, level)
    if crossings.size < 2:
        raise ValueError("Not enough zero-crossing points found")
    return crossings[0], level, crossings[-1], level


@check_1d_arrays(x_sorted=True)
def fwhm(
    x: np.ndarray,
    y: np.ndarray,
    method: Literal["zero-crossing", "gauss", "lorentz", "voigt"] = "zero-crossing",
    xmin: float | None = None,
    xmax: float | None = None,
) -> tuple[float, float, float, float]:
    """Compute Full Width at Half Maximum (FWHM) of the input data

    Args:
        x: X data
        y: Y data
        method: Calculation method. Two types of methods are supported: a zero-crossing
         method and fitting methods (based on various models: Gauss, Lorentz, Voigt).
         Defaults to "zero-crossing".
        xmin: Lower X bound for the fitting. Defaults to None (no lower bound,
         i.e. the fitting starts from the first point).
        xmax: Upper X bound for the fitting. Defaults to None (no upper bound,
         i.e. the fitting ends at the last point)

    Returns:
        FWHM segment coordinates
    """
    dx, dy, base = np.max(x) - np.min(x), np.max(y) - np.min(y), np.min(y)
    sigma, mu = dx * 0.1, peakdetection.xpeak(x, y)
    if isinstance(xmin, float):
        indices = np.where(x >= xmin)[0]
        x = x[indices]
        y = y[indices]
    if isinstance(xmax, float):
        indices = np.where(x <= xmax)[0]
        x = x[indices]
        y = y[indices]

    if method == "zero-crossing":
        hmax = dy * 0.5 + np.min(y)
        fx = features.find_all_x_at_given_y_value(x, y, hmax)
        if fx.size > 2:
            warnings.warn(f"Ambiguous zero-crossing points (found {fx.size} points)")
        elif fx.size < 2:
            raise ValueError("No zero-crossing points found")
        return fx[0], hmax, fx[-1], hmax

    try:
        fit_model_class: type[PulseFitModel] = {
            "gauss": GaussianModel,
            "lorentz": LorentzianModel,
            "voigt": VoigtModel,
        }[method]
    except KeyError as exc:
        raise ValueError(f"Invalid method {method}") from exc

    def func(params):
        """Fitting model function"""
        # pylint: disable=cell-var-from-loop
        return y - fit_model_class.func(x, *params)

    amp = fit_model_class.get_amp_from_amplitude(dy, sigma)
    (amp, sigma, mu, base), _ier = scipy.optimize.leastsq(
        func, np.array([amp, sigma, mu, base])
    )
    return fit_model_class.half_max_segment(amp, sigma, mu, base)


@check_1d_arrays(x_sorted=True)
def fw1e2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Compute Full Width at 1/e² of the input data (using a Gaussian model fitting).

    Args:
        x: X data
        y: Y data

    Returns:
        FW at 1/e² segment coordinates
    """
    dx, dy, base = np.max(x) - np.min(x), np.max(y) - np.min(y), np.min(y)
    sigma, mu = dx * 0.1, peakdetection.xpeak(x, y)
    amp = GaussianModel.get_amp_from_amplitude(dy, sigma)
    p_in = np.array([amp, sigma, mu, base])

    def func(params):
        """Fitting model function"""
        # pylint: disable=cell-var-from-loop
        return y - GaussianModel.func(x, *params)

    p_out, _ier = scipy.optimize.leastsq(func, p_in)
    amp, sigma, mu, base = p_out
    hw = 2 * sigma
    yhm = GaussianModel.amplitude(amp, sigma) / np.e**2 + base
    return mu - hw, yhm, mu + hw, yhm
