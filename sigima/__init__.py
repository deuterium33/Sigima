# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Sigima
======

Sigima is a scientific computing engine for 1D signals and 2D images.

It provides a set of tools for image and signal processing, including
denoising, segmentation, and restoration. It is designed to be used in
scientific and research applications.

It is a part of the DataLab Platform, which aims at providing a
comprehensive set of tools for data analysis and visualization, around
the DataLab application.
"""

# TODO: Use `numpy.typing.NDArray` for more precise type annotations once NumPy >= 1.21
# can be safely required (e.g. after raising the minimum required version of
# scikit-image to >= 0.19).

# The following comments are used to track the migration process of the `sigima`
# package, in the context of the DataLab Core Architecture Redesign project funded by
# the NLnet Foundation.

# -------- Point of no return after creating an independent `sigima` package ----------
# ** Task 3. Documentation and Training Materials **
# TODO: Move DataLab's validation section to sigima's documentation
# TODO: Add documentation on I/O plugin system (similar to the `cdl.plugins` module)
# --------------------------------------------------------------------------------------

__all__ = [
    "NO_ROI",
    "CircularROI",
    "ExponentialParam",
    "Gauss2DParam",
    "GaussParam",
    "GeometryResult",
    "ImageDatatypes",
    "ImageObj",
    "ImageROI",
    "ImageTypes",
    "KindShape",
    "LinearChirpParam",
    "LogisticParam",
    "LorentzParam",
    "NormalDistribution1DParam",
    "NormalDistribution2DParam",
    "PlanckParam",
    "PolygonalROI",
    "ROI1DParam",
    "ROI2DParam",
    "Ramp2DParam",
    "RectangularROI",
    "SegmentROI",
    "SignalObj",
    "SignalROI",
    "SignalTypes",
    "SimpleBaseProxy",
    "SimpleRemoteProxy",
    "StepParam",
    "TableResult",
    "TypeObj",
    "TypeROI",
    "UniformDistribution1DParam",
    "UniformDistribution2DParam",
    "VoigtParam",
    "calc_table_from_data",
    "create_image",
    "create_image_from_param",
    "create_image_parameters",
    "create_image_roi",
    "create_signal",
    "create_signal_from_param",
    "create_signal_parameters",
    "create_signal_roi",
    "read_image",
    "read_images",
    "read_signal",
    "read_signals",
    "write_image",
    "write_signal",
]


from guidata.config import ValidationMode, set_validation_mode

from sigima.client import SimpleBaseProxy, SimpleRemoteProxy
from sigima.io import (
    read_image,
    read_images,
    read_signal,
    read_signals,
    write_image,
    write_signal,
)
from sigima.objects import (
    NO_ROI,
    CircularROI,
    ExponentialParam,
    Gauss2DParam,
    GaussParam,
    GeometryResult,
    ImageDatatypes,
    ImageObj,
    ImageROI,
    ImageTypes,
    KindShape,
    LinearChirpParam,
    LogisticParam,
    LorentzParam,
    NormalDistribution1DParam,
    NormalDistribution2DParam,
    PlanckParam,
    PolygonalROI,
    Ramp2DParam,
    RectangularROI,
    ROI1DParam,
    ROI2DParam,
    SegmentROI,
    SignalObj,
    SignalROI,
    SignalTypes,
    StepParam,
    TableResult,
    TypeObj,
    TypeROI,
    UniformDistribution1DParam,
    UniformDistribution2DParam,
    VoigtParam,
    calc_table_from_data,
    create_image,
    create_image_from_param,
    create_image_parameters,
    create_image_roi,
    create_signal,
    create_signal_from_param,
    create_signal_parameters,
    create_signal_roi,
)

# Set validation mode to ENABLED by default (issue warnings for invalid inputs)
set_validation_mode(ValidationMode.ENABLED)

__version__ = "1.0.0b1"
__docurl__ = "https://sigima.readthedocs.io/en/latest/"
__homeurl__ = "https://github.com/DataLab-Platform/Sigima"
__supporturl__ = "https://github.com/DataLab-Platform/sigima/issues/new/choose"

# Dear (Debian, RPM, ...) package makers, please feel free to customize the
# following path to module's data (images) and translations:
DATAPATH = LOCALEPATH = ""
