# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Model classes for signals and images (:mod:`sigima.objects`)
------------------------------------------------------------

The :mod:`sigima.objects` module aims at providing all the necessary classes and
functions to create and manipulate Sigima signal and image objects.

Those classes and functions are defined in submodules:

- :mod:`sigima.objects.base`
- :mod:`sigima.objects.image`
- :mod:`sigima.objects.signal`

.. code-block:: python

    # Full import statement
    from sigima.objects.signal import SignalObj
    from sigima.objects.image import ImageObj

    # Short import statement
    from sigima.objects import SignalObj, ImageObj

Common objects
^^^^^^^^^^^^^^

.. autoclass:: sigima.objects.ResultProperties
    :members:
.. autoclass:: sigima.objects.ResultShape
    :members:
.. autoclass:: sigima.objects.ShapeTypes
    :members:
.. autoclass:: sigima.objects.TypeObj
.. autoclass:: sigima.objects.TypeROI
.. autoclass:: sigima.objects.TypeROIParam
.. autoclass:: sigima.objects.TypeSingleROI

Signal model
^^^^^^^^^^^^

.. autodataset:: sigima.objects.SignalObj
    :members:
    :inherited-members:
.. autofunction:: sigima.objects.create_signal_roi
.. autofunction:: sigima.objects.create_signal
.. autofunction:: sigima.objects.create_signal_from_param
.. autoclass:: sigima.objects.SignalTypes
.. autoclass:: sigima.objects.NewSignalParam
.. autodataset:: sigima.objects.ZerosParam
.. autodataset:: sigima.objects.UniformRandomParam
.. autodataset:: sigima.objects.NormalRandomParam
.. autodataset:: sigima.objects.GaussParam
.. autodataset:: sigima.objects.LorentzParam
.. autodataset:: sigima.objects.VoigtParam
.. autodataset:: sigima.objects.SinusParam
.. autodataset:: sigima.objects.CosinusParam
.. autodataset:: sigima.objects.SawtoothParam
.. autodataset:: sigima.objects.TriangleParam
.. autodataset:: sigima.objects.SquareParam
.. autodataset:: sigima.objects.SincParam
.. autodataset:: sigima.objects.StepParam
.. autodataset:: sigima.objects.ExponentialParam
.. autodataset:: sigima.objects.PulseParam
.. autodataset:: sigima.objects.PolyParam
.. autodataset:: sigima.objects.CustomSignalParam
.. autodataset:: sigima.objects.ROI1DParam
.. autoclass:: sigima.objects.SignalROI

Image model
^^^^^^^^^^^

.. autodataset:: sigima.objects.ImageObj
    :members:
    :inherited-members:
.. autofunction:: sigima.objects.create_image_roi
.. autofunction:: sigima.objects.create_image
.. autofunction:: sigima.objects.create_image_from_param
.. autoclass:: sigima.objects.ImageTypes
.. autoclass:: sigima.objects.NewImageParam
.. autodataset:: sigima.objects.Zeros2DParam
.. autodataset:: sigima.objects.Empty2DParam
.. autodataset:: sigima.objects.UniformRandom2DParam
.. autodataset:: sigima.objects.NormalRandom2DParam
.. autodataset:: sigima.objects.Gauss2DParam
.. autodataset:: sigima.objects.Ramp2DParam
.. autodataset:: sigima.objects.ROI2DParam
.. autoclass:: sigima.objects.ImageROI
.. autoclass:: sigima.objects.ImageDatatypes
"""

# pylint:disable=unused-import
# flake8: noqa

from sigima.objects.base import (
    ResultProperties,
    ResultShape,
    ShapeTypes,
    TypeObj,
    TypeROI,
    TypeROIParam,
    TypeSingleROI,
)
from sigima.objects.image import (
    CircularROI,
    Empty2DParam,
    Gauss2DParam,
    ImageDatatypes,
    ImageObj,
    ImageROI,
    ImageTypes,
    NewImageParam,
    NormalRandom2DParam,
    PolygonalROI,
    Ramp2DParam,
    RectangularROI,
    ROI2DParam,
    UniformRandom2DParam,
    Zeros2DParam,
    create_image,
    create_image_from_param,
    create_image_roi,
    get_image_parameters_class,
)
from sigima.objects.signal import (
    CosinusParam,
    CustomSignalParam,
    ExponentialParam,
    GaussParam,
    LorentzParam,
    NewSignalParam,
    NormalRandomParam,
    PolyParam,
    PulseParam,
    ROI1DParam,
    SawtoothParam,
    SegmentROI,
    SignalObj,
    SignalROI,
    SignalTypes,
    SincParam,
    SinusParam,
    SquareParam,
    StepParam,
    TriangleParam,
    UniformRandomParam,
    VoigtParam,
    ZerosParam,
    create_signal,
    create_signal_from_param,
    create_signal_roi,
    get_signal_parameters_class,
)
