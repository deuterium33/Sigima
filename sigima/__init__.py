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

# TODO: Should we use `ImageROI` instead of `ROI2DParam` as input parameter type
# for the `erase`, `extract_roi`, and `extract_rois` functions? (same for signal
# ROI extraction functions)

# TODO: Should we use an unified approach for handling enum types or similar choices?
# For example, see `BINNING_OPERATIONS` which currently is a simple tuple of strings,
# whereas it could be a more structured enum type. There are several other examples
# in the codebase where choices are defined as tuples or tuples of tuples instead of
# using a dedicated enum class.

# The following comments are used to track the migration process of the `sigima`
# package, in the context of the DataLab Core Architecture Redesign project funded by
# the NLnet Foundation.

# -------- Point of no return after creating an independent `sigima` package ----------
# TODO: Fix TODO related to `OPTIONS_RST` in 'sigima\config.py'
# TODO: Migrate `cdlclient` features to a subpackage of `sigima` (e.g., `sigima.client`)
#
# ** Task 3. Documentation and Training Materials **
# TODO: Move DataLab's validation section to sigima's documentation
# TODO: Add documentation on I/O plugin system (similar to the `cdl.plugins` module)
# --------------------------------------------------------------------------------------

# pylint:disable=unused-import
# flake8: noqa

from sigima.io import (
    read_image,
    read_images,
    read_signal,
    read_signals,
    write_image,
    write_signal,
)
from sigima.objects import (
    CircularROI,
    ExponentialParam,
    Gauss2DParam,
    GaussParam,
    ImageDatatypes,
    ImageObj,
    ImageROI,
    ImageTypes,
    LinearChirpParam,
    LogisticParam,
    LorentzParam,
    NormalRandom2DParam,
    NormalRandomParam,
    PlanckParam,
    PolygonalROI,
    Ramp2DParam,
    RectangularROI,
    ResultProperties,
    ResultShape,
    ROI1DParam,
    ROI2DParam,
    SegmentROI,
    ShapeTypes,
    SignalObj,
    SignalROI,
    SignalTypes,
    StepParam,
    TypeObj,
    TypeROI,
    UniformRandom2DParam,
    UniformRandomParam,
    VoigtParam,
    create_image,
    create_image_from_param,
    create_image_roi,
    create_signal,
    create_signal_from_param,
    create_signal_roi,
)
from guidata.config import ValidationMode, set_validation_mode

# Set validation mode to ENABLED by default (issue warnings for invalid inputs)
set_validation_mode(ValidationMode.ENABLED)

__version__ = "0.3.0"
__docurl__ = "https://sigima.readthedocs.io/en/latest/"
__homeurl__ = "https://github.com/DataLab-Platform/Sigima"
__supporturl__ = "https://github.com/DataLab-Platform/sigima/issues/new/choose"

# Dear (Debian, RPM, ...) package makers, please feel free to customize the
# following path to module's data (images) and translations:
DATAPATH = LOCALEPATH = ""
