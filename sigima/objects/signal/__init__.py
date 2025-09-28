# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Signal objects subpackage
=========================

This subpackage provides signal data structures and utilities.

The subpackage is organized into the following modules:

- `roi`: Region of Interest (ROI) classes and parameters for 1D signals
- `object`: Main SignalObj class for handling 1D signal data
- `creation`: Signal creation utilities and parameter classes

All classes and functions are re-exported at the subpackage level for backward
compatibility. Existing imports like `from sigima.objects.signal import SignalObj`
will continue to work.
"""

# Import all public classes and functions from submodules
from .creation import (
    # Constants
    DEFAULT_TITLE,
    # Mathematical function parameter classes
    BaseGaussLorentzVoigtParam,
    BasePeriodicParam,
    BasePulseParam,
    CosinusParam,
    CustomSignalParam,
    # Pulse signal classes
    ExpectedFeatures,
    ExponentialParam,
    FeatureTolerances,
    # Periodic function parameter classes
    FreqUnits,
    GaussParam,
    # Other signal types
    LinearChirpParam,
    LogisticParam,
    LorentzParam,
    # Base parameter classes
    NewSignalParam,
    NormalDistribution1DParam,
    PlanckParam,
    PoissonDistribution1DParam,
    # Polynomial and custom signals
    PolyParam,
    PulseParam,
    SawtoothParam,
    # Enums
    SignalTypes,
    SincParam,
    SinusParam,
    SquareParam,
    SquarePulseParam,
    StepParam,
    StepPulseParam,
    TriangleParam,
    UniformDistribution1DParam,
    VoigtParam,
    # Distribution parameter classes
    ZerosParam,
    check_all_signal_parameters_classes,
    # Core creation functions
    create_signal,
    create_signal_from_param,
    # Factory and utility functions
    create_signal_parameters,
    get_next_signal_number,
    # Registration functions
    register_signal_parameters_class,
    triangle_func,
)
from .object import (
    # Main signal class
    SignalObj,
)
from .roi import (
    # ROI classes
    ROI1DParam,
    SegmentROI,
    SignalROI,
    # ROI functions
    create_signal_roi,
)

# Define __all__ for explicit public API
__all__ = [
    # From roi module
    "ROI1DParam",
    "SegmentROI",
    "SignalROI",
    "create_signal_roi",
    # From object module
    "SignalObj",
    "create_signal",
    # From creation module
    "create_signal_from_param",
    "SignalTypes",
    "NewSignalParam",
    "ZerosParam",
    "UniformDistribution1DParam",
    "NormalDistribution1DParam",
    "PoissonDistribution1DParam",
    "BaseGaussLorentzVoigtParam",
    "GaussParam",
    "LorentzParam",
    "VoigtParam",
    "PlanckParam",
    "FreqUnits",
    "BasePeriodicParam",
    "SinusParam",
    "CosinusParam",
    "SawtoothParam",
    "TriangleParam",
    "SquareParam",
    "SincParam",
    "LinearChirpParam",
    "StepParam",
    "ExponentialParam",
    "LogisticParam",
    "PulseParam",
    "ExpectedFeatures",
    "FeatureTolerances",
    "BasePulseParam",
    "StepPulseParam",
    "SquarePulseParam",
    "PolyParam",
    "CustomSignalParam",
    "create_signal_parameters",
    "get_next_signal_number",
    "triangle_func",
    "register_signal_parameters_class",
    "check_all_signal_parameters_classes",
    "DEFAULT_TITLE",
]
