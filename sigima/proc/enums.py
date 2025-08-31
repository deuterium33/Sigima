"""Common enum definitions for Sigima processing."""

from __future__ import annotations

from enum import Enum


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
