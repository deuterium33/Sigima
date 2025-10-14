# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image objects subpackage
========================

This subpackage provides image data structures and utilities.

The subpackage is organized into the following modules:

- `roi`: Region of Interest (ROI) classes and parameters
- `object`: Main ImageObj class for handling 2D image data
- `creation`: Image creation utilities and parameter classes

All classes and functions are re-exported at the subpackage level for backward
compatibility. Existing imports like `from sigima.objects.image import ImageObj`
will continue to work.
"""

# Import all public classes and functions from submodules
from .creation import (
    # Constants
    DEFAULT_TITLE,
    Gauss2DParam,
    # Enums
    ImageDatatypes,
    ImageTypes,
    # Base parameter classes
    NewImageParam,
    NormalDistribution2DParam,
    PoissonDistribution2DParam,
    Ramp2DParam,
    UniformDistribution2DParam,
    # Specific parameter classes
    Zero2DParam,
    check_all_image_parameters_classes,
    # Core creation function
    create_image,
    create_image_from_param,
    # Factory and utility functions
    create_image_parameters,
    get_next_image_number,
    # Registration functions
    register_image_parameters_class,
)
from .object import (
    ImageObj,
)
from .roi import (
    # ROI classes
    BaseSingleImageROI,
    CircularROI,
    # Specific ROI types
    ImageROI,
    PolygonalROI,
    RectangularROI,
    ROI2DParam,
    # ROI utility function
    create_image_roi,
)

# Define __all__ for explicit public API
__all__ = [
    # From roi module
    "ROI2DParam",
    "BaseSingleImageROI",
    "PolygonalROI",
    "RectangularROI",
    "CircularROI",
    "ImageROI",
    "create_image_roi",
    # From object module
    "ImageObj",
    # From creation module
    "create_image",
    "ImageDatatypes",
    "ImageTypes",
    "NewImageParam",
    "Zero2DParam",
    "UniformDistribution2DParam",
    "NormalDistribution2DParam",
    "PoissonDistribution2DParam",
    "Gauss2DParam",
    "Ramp2DParam",
    "create_image_parameters",
    "create_image_from_param",
    "get_next_image_number",
    "register_image_parameters_class",
    "check_all_image_parameters_classes",
    "DEFAULT_TITLE",
]
