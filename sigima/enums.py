# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Common enum definitions for Sigima processing."""

from __future__ import annotations

from enum import Enum

from sigima.config import _


class AngleUnit(Enum):
    """Angle units."""

    RADIAN = "rad"
    DEGREE = "°"


class BinningOperation(Enum):
    """Binning operations for image processing."""

    SUM = "sum"
    AVERAGE = "average"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"


class ContourShape(Enum):
    """Contour shapes for image processing."""

    ELLIPSE = _("Ellipse")
    CIRCLE = _("Circle")
    POLYGON = _("Polygon")


class BorderMode(Enum):
    """Border modes for filtering and image processing."""

    CONSTANT = "constant"
    NEAREST = "nearest"
    REFLECT = "reflect"
    WRAP = "wrap"
    MIRROR = "mirror"


class MathOperator(Enum):
    """Mathematical operators for data operations."""

    ADD = "+"
    SUBTRACT = "-"
    MULTIPLY = "×"
    DIVIDE = "/"


class FilterMode(Enum):
    """Filter modes for signal and image processing."""

    REFLECT = "reflect"
    CONSTANT = "constant"
    NEAREST = "nearest"
    MIRROR = "mirror"
    WRAP = "wrap"


class WaveletMode(Enum):
    """Wavelet transform modes."""

    CONSTANT = "constant"
    EDGE = "edge"
    SYMMETRIC = "symmetric"
    REFLECT = "reflect"
    WRAP = "wrap"


class ThresholdMethod(Enum):
    """Thresholding methods for wavelet denoising."""

    SOFT = "soft"
    HARD = "hard"


class ShrinkageMethod(Enum):
    """Shrinkage methods for wavelet denoising."""

    BAYES_SHRINK = "BayesShrink"
    VISU_SHRINK = "VisuShrink"


class PadLocation(Enum):
    """Padding location for signal processing."""

    APPEND = "append"
    PREPEND = "prepend"
    BOTH = "both"


class PowerUnit(Enum):
    """Power spectral density units."""

    DBC = "dBc"
    DBFS = "dBFS"


class WindowingMethod(Enum):
    """Windowing methods enumeration."""

    BARTHANN = "Barthann"
    BARTLETT = "Bartlett"
    BLACKMAN = "Blackman"
    BLACKMAN_HARRIS = "Blackman-Harris"
    BOHMAN = "Bohman"
    BOXCAR = "Boxcar"
    COSINE = _("Cosine")
    EXPONENTIAL = _("Exponential")
    FLAT_TOP = _("Flat Top")
    GAUSSIAN = _("Gaussian")
    HAMMING = "Hamming"
    HANN = "Hann"
    KAISER = "Kaiser"
    LANCZOS = "Lanczos"
    NUTTALL = "Nuttall"
    PARZEN = "Parzen"
    TAYLOR = "Taylor"
    TUKEY = "Tukey"


class Interpolation1DMethod(Enum):
    """Methods for 1D interpolation and resampling."""

    LINEAR = _("Linear")
    SPLINE = _("Spline")
    QUADRATIC = _("Quadratic")
    CUBIC = _("Cubic")
    BARYCENTRIC = _("Barycentric")
    PCHIP = _("PCHIP")


class Interpolation2DMethod(Enum):
    """Methods for 2D interpolation and resampling."""

    NEAREST = _("Nearest")
    LINEAR = _("Linear")
    CUBIC = _("Cubic")


class NormalizationMethod(Enum):
    """Normalization methods for signal processing."""

    MAXIMUM = _("Maximum")
    AMPLITUDE = _("Amplitude")
    AREA = _("Area")
    ENERGY = _("Energy")
    RMS = _("RMS")


class FilterType(Enum):
    """Filter types"""

    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"
    BANDSTOP = "bandstop"


class FrequencyFilterMethod(Enum):
    """Frequency filter methods for signal processing."""

    BESSEL = "Bessel"
    BRICKWALL = _("Brickwall")
    BUTTERWORTH = "Butterworth"
    CHEBYSHEV1 = "Chebyshev I"
    CHEBYSHEV2 = "Chebyshev II"
    ELLIPTIC = _("Elliptic")


class SignalShape(str, Enum):
    """Enum for signal shapes.

    WARNING: This enum inherits from str and uses raw string values (not translated).
    This is intentional and differs from other enums in this module for the following
    reasons:

    - SignalShape values are used for data serialization (JSON, database storage)
    - Values must be language-independent and stable across translations
    - The str inheritance enables automatic JSON serialization without manual conversion
    - For UI display, use a separate translation mechanism rather than enum values

    Compare with other enums (ContourShape, WindowingMethod, etc.) that use _("...")
    for translated values since they are primarily for user interface display.

    Example usage:
        shape = SignalShape.STEP
        json_data = {"shape": shape}  # Serializes as {"shape": "step"}
        # For display: implement a separate display_name property or translation
    """

    STEP = "step"
    SQUARE = "square"
