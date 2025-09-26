# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Common enum definitions for Sigima processing."""

from __future__ import annotations

from enum import Enum

import guidata.dataset as gds

from sigima.config import _


class AngleUnit(gds.LabeledEnum):
    """Angle units."""

    RADIAN = "rad"
    DEGREE = "°"


class BinningOperation(gds.LabeledEnum):
    """Binning operations for image processing."""

    SUM = "sum"
    AVERAGE = "average"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"


class ContourShape(gds.LabeledEnum):
    """Contour shapes for image processing."""

    ELLIPSE = "ellipse", _("Ellipse")
    CIRCLE = "circle", _("Circle")
    POLYGON = "polygon", _("Polygon")


class BorderMode(gds.LabeledEnum):
    """Border modes for filtering and image processing."""

    CONSTANT = "constant"
    NEAREST = "nearest"
    REFLECT = "reflect"
    WRAP = "wrap"
    MIRROR = "mirror"


class MathOperator(gds.LabeledEnum):
    """Mathematical operators for data operations."""

    ADD = "+"
    SUBTRACT = "-"
    MULTIPLY = "×"
    DIVIDE = "/"


class FilterMode(gds.LabeledEnum):
    """Filter modes for signal and image processing."""

    REFLECT = "reflect"
    CONSTANT = "constant"
    NEAREST = "nearest"
    MIRROR = "mirror"
    WRAP = "wrap"


class WaveletMode(gds.LabeledEnum):
    """Wavelet transform modes."""

    CONSTANT = "constant"
    EDGE = "edge"
    SYMMETRIC = "symmetric"
    REFLECT = "reflect"
    WRAP = "wrap"


class ThresholdMethod(gds.LabeledEnum):
    """Thresholding methods for wavelet denoising."""

    SOFT = "soft"
    HARD = "hard"


class ShrinkageMethod(gds.LabeledEnum):
    """Shrinkage methods for wavelet denoising."""

    BAYES_SHRINK = "BayesShrink"
    VISU_SHRINK = "VisuShrink"


class PadLocation(gds.LabeledEnum):
    """Padding location for signal processing."""

    APPEND = "append"
    PREPEND = "prepend"
    BOTH = "both"


class PowerUnit(gds.LabeledEnum):
    """Power spectral density units."""

    DBC = "dBc"
    DBFS = "dBFS"


class WindowingMethod(gds.LabeledEnum):
    """Windowing methods enumeration."""

    BARTHANN = "barthann", "Barthann"
    BARTLETT = "bartlett", "Bartlett"
    BLACKMAN = "blackman", "Blackman"
    BLACKMAN_HARRIS = "blackman_harris", "Blackman-Harris"
    BOHMAN = "bohman", "Bohman"
    BOXCAR = "boxcar", "Boxcar"
    COSINE = "cosine", _("Cosine")
    EXPONENTIAL = "exponential", _("Exponential")
    FLAT_TOP = "flat_top", _("Flat Top")
    GAUSSIAN = "gaussian", _("Gaussian")
    HAMMING = "hamming", "Hamming"
    HANN = "hann", "Hann"
    KAISER = "kaiser", "Kaiser"
    LANCZOS = "lanczos", "Lanczos"
    NUTTALL = "nuttall", "Nuttall"
    PARZEN = "parzen", "Parzen"
    TAYLOR = "taylor", "Taylor"
    TUKEY = "tukey", "Tukey"


class Interpolation1DMethod(gds.LabeledEnum):
    """Methods for 1D interpolation and resampling."""

    LINEAR = "linear", _("Linear")
    SPLINE = "spline", _("Spline")
    QUADRATIC = "quadratic", _("Quadratic")
    CUBIC = "cubic", _("Cubic")
    BARYCENTRIC = "barycentric", _("Barycentric")
    PCHIP = "pchip", _("PCHIP")


class Interpolation2DMethod(gds.LabeledEnum):
    """Methods for 2D interpolation and resampling."""

    NEAREST = "nearest", _("Nearest")
    LINEAR = "linear", _("Linear")
    CUBIC = "cubic", _("Cubic")


class NormalizationMethod(gds.LabeledEnum):
    """Normalization methods for signal processing."""

    MAXIMUM = "maximum", _("Maximum")
    AMPLITUDE = "amplitude", _("Amplitude")
    AREA = "area", _("Area")
    ENERGY = "energy", _("Energy")
    RMS = "rms", _("RMS")


class FilterType(gds.LabeledEnum):
    """Filter types"""

    LOWPASS = "lowpass", "lowpass"
    HIGHPASS = "highpass", "highpass"
    BANDPASS = "bandpass", "bandpass"
    BANDSTOP = "bandstop", "bandstop"


class FrequencyFilterMethod(gds.LabeledEnum):
    """Frequency filter methods for signal processing."""

    BESSEL = "bessel", "Bessel"
    BRICKWALL = "brickwall", _("Brickwall")
    BUTTERWORTH = "butterworth", "Butterworth"
    CHEBYSHEV1 = "chebyshev1", "Chebyshev I"
    CHEBYSHEV2 = "chebyshev2", "Chebyshev II"
    ELLIPTIC = "elliptic", _("Elliptic")


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
