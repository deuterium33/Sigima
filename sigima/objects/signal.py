# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Signal object and related classes
---------------------------------

"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

import enum
from typing import Type

import guidata.dataset as gds
import numpy as np
import scipy.constants
import scipy.signal as sps

from sigima.config import _
from sigima.objects import base
from sigima.tools.signal.pulse import GaussianModel, LorentzianModel, VoigtModel


class ROI1DParam(base.BaseROIParam["SignalObj", "SegmentROI"]):
    """Signal ROI parameters"""

    # Note: in this class, the ROI parameters are stored as X coordinates

    title = gds.StringItem(_("ROI title"), default="")
    xmin = gds.FloatItem(_("First point coordinate"), default=0.0)
    xmax = gds.FloatItem(_("Last point coordinate"), default=1.0)

    def to_single_roi(self, obj: SignalObj) -> SegmentROI:
        """Convert parameters to single ROI

        Args:
            obj: signal object

        Returns:
            Single ROI
        """
        assert isinstance(self.xmin, float) and isinstance(self.xmax, float)
        return SegmentROI([self.xmin, self.xmax], False, title=self.title)

    def get_data(self, obj: SignalObj) -> np.ndarray:
        """Get signal data in ROI

        Args:
            obj: signal object

        Returns:
            Data in ROI
        """
        assert isinstance(self.xmin, float) and isinstance(self.xmax, float)
        imin, imax = np.searchsorted(obj.x, [self.xmin, self.xmax])
        return np.array([obj.x[imin:imax], obj.y[imin:imax]])


class SegmentROI(base.BaseSingleROI["SignalObj", ROI1DParam]):
    """Segment ROI

    Args:
        coords: ROI coordinates (xmin, xmax)
        title: ROI title
    """

    # Note: in this class, the ROI parameters are stored as X indices

    def check_coords(self) -> None:
        """Check if coords are valid

        Raises:
            ValueError: invalid coords
        """
        if len(self.coords) != 2:
            raise ValueError("Invalid ROI segment coords (2 values expected)")
        if self.coords[0] >= self.coords[1]:
            raise ValueError("Invalid ROI segment coords (xmin >= xmax)")

    def get_data(self, obj: SignalObj) -> tuple[np.ndarray, np.ndarray]:
        """Get signal data in ROI

        Args:
            obj: signal object

        Returns:
            Data in ROI
        """
        imin, imax = self.get_indices_coords(obj)
        return obj.x[imin:imax], obj.y[imin:imax]

    def to_mask(self, obj: SignalObj) -> np.ndarray:
        """Create mask from ROI

        Args:
            obj: signal object

        Returns:
            Mask (boolean array where True values are inside the ROI)
        """
        mask = np.ones_like(obj.xydata, dtype=bool)
        imin, imax = self.get_indices_coords(obj)
        mask[:, imin:imax] = False
        return mask

    # pylint: disable=unused-argument
    def to_param(self, obj: SignalObj, index: int) -> ROI1DParam:
        """Convert ROI to parameters

        Args:
            obj: object (signal), for physical-indices coordinates conversion
            index: ROI index
        """
        gtitle = base.get_generic_roi_title(index)
        param = ROI1DParam(gtitle)
        param.title = self.title or gtitle
        param.xmin, param.xmax = self.get_physical_coords(obj)
        return param


class SignalROI(base.BaseROI["SignalObj", SegmentROI, ROI1DParam]):
    """Signal Regions of Interest

    Args:
        inverse: if True, ROI is outside the region
    """

    PREFIX = "s"

    @staticmethod
    def get_compatible_single_roi_classes() -> list[Type[SegmentROI]]:
        """Return compatible single ROI classes"""
        return [SegmentROI]

    def to_mask(self, obj: SignalObj) -> np.ndarray:
        """Create mask from ROI

        Args:
            obj: signal object

        Returns:
            Mask (boolean array where True values are inside the ROI)
        """
        mask = np.ones_like(obj.xydata, dtype=bool)
        if self.single_rois:
            for roi in self.single_rois:
                mask &= roi.to_mask(obj)
        else:
            # If no single ROIs, the mask is empty (no ROI defined)
            mask[:] = False
        return mask


def create_signal_roi(
    coords: np.ndarray | list[float] | list[list[float]],
    indices: bool = False,
    inverse: bool = False,
    title: str = "",
) -> SignalROI:
    """Create Signal Regions of Interest (ROI) object.
    More ROIs can be added to the object after creation, using the `add_roi` method.

    Args:
        coords: single ROI coordinates `[xmin, xmax]`, or multiple ROIs coordinates
         `[[xmin1, xmax1], [xmin2, xmax2], ...]` (lists or NumPy arrays)
        indices: if True, coordinates are indices, if False, they are physical values
         (default to False for signals)
        inverse: if True, ROI is outside the region
        title: title

    Returns:
        Regions of Interest (ROI) object

    Raises:
        ValueError: if the number of coordinates is not even
    """
    coords = np.array(coords, float)
    if coords.ndim == 1:
        coords = coords.reshape(1, -1)
    roi = SignalROI(inverse)
    for row in coords:
        roi.add_roi(SegmentROI(row, indices=indices, title=title))
    return roi


class SignalObj(gds.DataSet, base.BaseObj[SignalROI]):
    """Signal object"""

    PREFIX = "s"
    VALID_DTYPES = (np.float32, np.float64, np.complex128)

    _tabs = gds.BeginTabGroup("all")

    _datag = gds.BeginGroup(_("Data and metadata"))
    title = gds.StringItem(_("Signal title"), default=_("Untitled"))
    xydata = gds.FloatArrayItem(_("Data"), transpose=True, minmax="rows")
    metadata = gds.DictItem(_("Metadata"), default={})  # type: ignore[assignment]
    annotations = gds.StringItem(_("Annotations"), default="").set_prop(
        "display",
        hide=True,
    )  # Annotations as a serialized JSON string  # type: ignore[assignment]
    _e_datag = gds.EndGroup(_("Data and metadata"))

    _unitsg = gds.BeginGroup(_("Titles and units"))
    title = gds.StringItem(_("Signal title"), default=_("Untitled"))
    _tabs_u = gds.BeginTabGroup("units")
    _unitsx = gds.BeginGroup(_("X-axis"))
    xlabel = gds.StringItem(_("Title"), default="")
    xunit = gds.StringItem(_("Unit"), default="")
    _e_unitsx = gds.EndGroup(_("X-axis"))
    _unitsy = gds.BeginGroup(_("Y-axis"))
    ylabel = gds.StringItem(_("Title"), default="")
    yunit = gds.StringItem(_("Unit"), default="")
    _e_unitsy = gds.EndGroup(_("Y-axis"))
    _e_tabs_u = gds.EndTabGroup("units")
    _e_unitsg = gds.EndGroup(_("Titles and units"))

    _scalesg = gds.BeginGroup(_("Scales"))
    _prop_autoscale = gds.GetAttrProp("autoscale")
    autoscale = gds.BoolItem(_("Auto scale"), default=True).set_prop(
        "display", store=_prop_autoscale
    )
    _tabs_b = gds.BeginTabGroup("bounds")
    _boundsx = gds.BeginGroup(_("X-axis"))
    xscalelog = gds.BoolItem(_("Logarithmic scale"), default=False)
    xscalemin = gds.FloatItem(_("Lower bound"), check=False).set_prop(
        "display", active=gds.NotProp(_prop_autoscale)
    )
    xscalemax = gds.FloatItem(_("Upper bound"), check=False).set_prop(
        "display", active=gds.NotProp(_prop_autoscale)
    )
    _e_boundsx = gds.EndGroup(_("X-axis"))
    _boundsy = gds.BeginGroup(_("Y-axis"))
    yscalelog = gds.BoolItem(_("Logarithmic scale"), default=False)
    yscalemin = gds.FloatItem(_("Lower bound"), check=False).set_prop(
        "display", active=gds.NotProp(_prop_autoscale)
    )
    yscalemax = gds.FloatItem(_("Upper bound"), check=False).set_prop(
        "display", active=gds.NotProp(_prop_autoscale)
    )
    _e_boundsy = gds.EndGroup(_("Y-axis"))
    _e_tabs_b = gds.EndTabGroup("bounds")
    _e_scalesg = gds.EndGroup(_("Scales"))

    _e_tabs = gds.EndTabGroup("all")

    def __init__(self, title=None, comment=None, icon=""):
        """Constructor

        Args:
            title: title
            comment: comment
            icon: icon
        """
        gds.DataSet.__init__(self, title, comment, icon)
        base.BaseObj.__init__(self)

    @staticmethod
    def get_roi_class() -> Type[SignalROI]:
        """Return ROI class"""
        return SignalROI

    def copy(
        self, title: str | None = None, dtype: np.dtype | None = None
    ) -> SignalObj:
        """Copy object.

        Args:
            title: title
            dtype: data type

        Returns:
            Copied object
        """
        title = self.title if title is None else title
        obj = SignalObj(title=title)
        obj.title = title
        obj.xlabel = self.xlabel
        obj.xunit = self.xunit
        obj.yunit = self.yunit
        if dtype not in (None, float, complex, np.complex128):
            raise RuntimeError("Signal data only supports float64/complex128 dtype")
        obj.metadata = base.deepcopy_metadata(self.metadata)
        obj.annotations = self.annotations
        obj.xydata = np.array(self.xydata, copy=True, dtype=dtype)
        return obj

    def set_data_type(self, dtype: np.dtype) -> None:  # pylint: disable=unused-argument
        """Change data type.

        Args:
            Data type
        """
        raise RuntimeError("Setting data type is not support for signals")

    def set_xydata(
        self,
        x: np.ndarray | list | None,
        y: np.ndarray | list | None,
        dx: np.ndarray | list | None = None,
        dy: np.ndarray | list | None = None,
    ) -> None:
        """Set xy data

        Args:
            x: x data
            y: y data
            dx: dx data (optional: error bars). Use None to reset dx data to None,
             or provide array to set new dx data.
            dy: dy data (optional: error bars). Use None to reset dy data to None,
             or provide array to set new dy data.
        """
        if x is None and y is None:
            # Using empty arrays (this allows initialization of the object without data)
            x = np.array([], dtype=np.float64)
            y = np.array([], dtype=np.float64)
        if x is None and y is not None:
            # If x is None, we create a default x array based on the length of y
            assert isinstance(y, (list, np.ndarray))
            x = np.arange(len(y), dtype=np.float64)
        if x is not None:
            x = np.array(x)
        if y is not None:
            y = np.array(y)
        if dx is not None:
            dx = np.array(dx)
        if dy is not None:
            dy = np.array(dy)
        if dx is None and dy is None:
            self.xydata = np.vstack([x, y])
        else:
            if dx is None:
                dx = np.full_like(x, np.nan)
            if dy is None:
                dy = np.full_like(y, np.nan)
            assert x is not None and y is not None
            self.xydata = np.vstack((x, y, dx, dy))

    def __get_x(self) -> np.ndarray | None:
        """Get x data"""
        if self.xydata is not None:
            x: np.ndarray = self.xydata[0]
            # We have to ensure that x is a floating point array, because if y is
            # complex, the whole xydata array will be complex, and we need to avoid
            # any unintended type promotion.
            return x.real.astype(float)
        return None

    def __set_x(self, data: np.ndarray | list[float]) -> None:
        """Set x data"""
        assert isinstance(self.xydata, np.ndarray)
        assert isinstance(data, (list, np.ndarray))
        data = np.array(data, dtype=float)
        assert data.shape[1] == self.xydata.shape[1], (
            "X data size must match Y data size"
        )
        if not np.all(np.diff(data) >= 0.0):
            raise ValueError("X data must be monotonic (sorted in ascending order)")
        self.xydata[0] = data

    def __get_y(self) -> np.ndarray | None:
        """Get y data"""
        if self.xydata is not None:
            return self.xydata[1]
        return None

    def __set_y(self, data: np.ndarray | list[float]) -> None:
        """Set y data"""
        assert isinstance(self.xydata, np.ndarray)
        assert isinstance(data, (list, np.ndarray))
        data = np.array(data)
        assert data.shape[0] == self.xydata.shape[1], (
            "Y data size must match X data size"
        )
        assert np.issubdtype(data.dtype, np.inexact), "Y data must be float or complex"
        self.xydata[1] = data

    def __get_dx(self) -> np.ndarray | None:
        """Get dx data"""
        if self.xydata is not None and len(self.xydata) == 4:
            dx: np.ndarray = self.xydata[2]
            if np.all(np.isnan(dx)):
                return None
            return dx.real.astype(float)
        return None

    def __set_dx(self, data: np.ndarray | list[float] | None) -> None:
        """Set dx data"""
        if data is None:
            data = np.full_like(self.x, np.nan)
        assert isinstance(data, (list, np.ndarray))
        data = np.array(data)
        if self.xydata is None:
            raise ValueError("Signal data not initialized")
        assert data.shape[0] == self.xydata.shape[1], (
            "dx data size must match X data size"
        )
        if len(self.xydata) == 2:
            self.xydata = np.vstack((self.xydata, np.zeros((2, self.xydata.shape[1]))))
        self.xydata[2] = np.array(data)

    def __get_dy(self) -> np.ndarray | None:
        """Get dy data"""
        if self.xydata is not None and len(self.xydata) == 4:
            dy: np.ndarray = self.xydata[3]
            if np.all(np.isnan(dy)):
                return None
            return dy
        return None

    def __set_dy(self, data: np.ndarray | list[float] | None) -> None:
        """Set dy data"""
        if data is None:
            data = np.full_like(self.x, np.nan)
        assert isinstance(data, (list, np.ndarray))
        data = np.array(data)
        if self.xydata is None:
            raise ValueError("Signal data not initialized")
        assert data.shape[0] == self.xydata.shape[1], (
            "dy data size must match X data size"
        )
        if len(self.xydata) == 2:
            self.xydata = np.vstack((self.xydata, np.zeros((2, self.xydata.shape[1]))))
        self.xydata[3] = np.array(data)

    x = property(__get_x, __set_x)
    y = data = property(__get_y, __set_y)
    dx = property(__get_dx, __set_dx)
    dy = property(__get_dy, __set_dy)

    def get_data(self, roi_index: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Return original data (if ROI is not defined or `roi_index` is None),
        or ROI data (if both ROI and `roi_index` are defined).

        Args:
            roi_index: ROI index

        Returns:
            Data
        """
        if self.roi is None or roi_index is None:
            assert isinstance(self.xydata, np.ndarray)
            return self.x, self.y
        single_roi = self.roi.get_single_roi(roi_index)
        return single_roi.get_data(self)

    def physical_to_indices(self, coords: list[float]) -> list[int]:
        """Convert coordinates from physical (real world) to indices (pixel)

        Args:
            coords: coordinates

        Returns:
            Indices
        """
        assert isinstance(self.x, np.ndarray)
        return [int(np.abs(self.x - x).argmin()) for x in coords]

    def indices_to_physical(self, indices: list[int]) -> list[float]:
        """Convert coordinates from indices to physical (real world)

        Args:
            indices: indices

        Returns:
            Coordinates
        """
        # We take the real part of the x data to avoid `ComplexWarning` warnings
        # when creating and manipulating the `XRangeSelection` shape (`plotpy`)
        return self.x.real[indices].tolist()


def create_signal(
    title: str,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    dx: np.ndarray | None = None,
    dy: np.ndarray | None = None,
    metadata: dict | None = None,
    units: tuple[str, str] | None = None,
    labels: tuple[str, str] | None = None,
) -> SignalObj:
    """Create a new Signal object.

    Args:
        title: signal title
        x: X data
        y: Y data
        dx: dX data (optional: error bars)
        dy: dY data (optional: error bars)
        metadata: signal metadata
        units: X, Y units (tuple of strings)
        labels: X, Y labels (tuple of strings)

    Returns:
        Signal object
    """
    assert isinstance(title, str)
    signal = SignalObj(title=title)
    signal.title = title
    signal.set_xydata(x, y, dx=dx, dy=dy)
    if units is not None:
        signal.xunit, signal.yunit = units
    if labels is not None:
        signal.xlabel, signal.ylabel = labels
    if metadata is not None:
        signal.metadata.update(metadata)
    return signal


class SignalTypes(enum.Enum):
    """Signal types"""

    #: Signal filled with zeros
    ZEROS = _("Zeros")
    #: Random signal (normal distribution)
    NORMAL_DISTRIBUTION = _("Normal distribution")
    #: Random signal (Poisson distribution)
    POISSON_DISTRIBUTION = _("Poisson distribution")
    #: Random signal (uniform distribution)
    UNIFORM_DISTRIBUTION = _("Uniform distribution")
    #: Gaussian function
    GAUSS = _("Gaussian")
    #: Lorentzian function
    LORENTZ = _("Lorentzian")
    #: Voigt function
    VOIGT = "Voigt"
    #: Planck function
    PLANCK = _("Blackbody (Planck)")
    #: Sinusoid
    SINUS = _("Sinus")
    #: Cosinusoid
    COSINUS = _("Cosinus")
    #: Sawtooth function
    SAWTOOTH = _("Sawtooth")
    #: Triangle function
    TRIANGLE = _("Triangle")
    #: Square function
    SQUARE = _("Square")
    #: Cardinal sine
    SINC = _("Cardinal sine")
    #: Linear chirp
    LINEARCHIRP = _("Linear chirp")
    #: Step function
    STEP = _("Step")
    #: Exponential function
    EXPONENTIAL = _("Exponential")
    #: Logistic function
    LOGISTIC = _("Logistic")
    #: Pulse function
    PULSE = _("Pulse")
    #: Polynomial function
    POLYNOMIAL = _("Polynomial")
    #: Custom function
    CUSTOM = _("Custom")


DEFAULT_TITLE = _("Untitled signal")


class NewSignalParam(gds.DataSet):
    """New signal dataset"""

    SIZE_RANGE_ACTIVATION_FLAG = True

    _size_range = gds.GetAttrProp("SIZE_RANGE_ACTIVATION_FLAG")
    title = gds.StringItem(_("Title"), default=DEFAULT_TITLE)
    size = gds.IntItem(
        _("N<sub>points</sub>"),
        help=_("Total number of points in the signal"),
        min=1,
        default=500,
    ).set_prop("display", active=_size_range)
    xmin = gds.FloatItem("x<sub>min</sub>", default=-10.0).set_prop(
        "display", active=_size_range
    )
    xmax = gds.FloatItem("x<sub>max</sub>", default=10.0).set_prop(
        "display", active=_size_range, col=1
    )
    xlabel = gds.StringItem(_("X label"), default="")
    ylabel = gds.StringItem(_("Y label"), default="").set_prop("display", col=1)
    xunit = gds.StringItem(_("X unit"), default="")
    yunit = gds.StringItem(_("Y unit"), default="").set_prop("display", col=1)

    # As it is the last item of the dataset, the separator will be hidden if no other
    # items are present after it (i.e. when derived classes do not add any new items
    # or when the NewSignalParam class is used alone).
    sep = gds.SeparatorItem()

    def generate_title(self) -> str:
        """Generate a title based on current parameters."""
        return ""

    def generate_x_data(self) -> np.ndarray:
        """Generate x data based on current parameters."""
        return np.linspace(self.xmin, self.xmax, self.size)

    def generate_1d_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1D data based on current parameters.

        Returns:
            Tuple of (x, y) arrays
        """
        return self.generate_x_data(), np.zeros(self.size)


SIGNAL_TYPE_PARAM_CLASSES = {}


def register_signal_parameters_class(stype: SignalTypes, param_class) -> None:
    """Register a parameters class for a given signal type.

    Args:
        stype: signal type
        param_class: parameters class
    """
    SIGNAL_TYPE_PARAM_CLASSES[stype] = param_class


def __get_signal_parameters_class(stype: SignalTypes) -> Type[NewSignalParam]:
    """Get parameters class for a given signal type.

    Args:
        stype: signal type

    Returns:
        Parameters class

    Raises:
        ValueError: if no parameters class is registered for the given signal type
    """
    try:
        return SIGNAL_TYPE_PARAM_CLASSES[stype]
    except KeyError as exc:
        raise ValueError(
            f"Image type {stype} has no parameters class registered"
        ) from exc


def check_all_signal_parameters_classes() -> None:
    """Check all registered parameters classes."""
    for stype, param_class in SIGNAL_TYPE_PARAM_CLASSES.items():
        assert __get_signal_parameters_class(stype) is param_class


def create_signal_parameters(
    stype: SignalTypes,
    title: str | None = None,
    xmin: float | None = None,
    xmax: float | None = None,
    size: int | None = None,
    **kwargs: dict,
) -> NewSignalParam:
    """Create parameters for a given signal type.

    Args:
        stype: signal type
        title: signal title
        xmin: minimum x value
        xmax: maximum x value
        size: signal size (number of points)
        **kwargs: additional parameters (specific to the signal type)

    Returns:
        Parameters object for the given signal type
    """
    pclass = __get_signal_parameters_class(stype)
    p = pclass.create(**kwargs)
    if title is not None:
        p.title = title
    if xmin is not None:
        p.xmin = xmin
    if xmax is not None:
        p.xmax = xmax
    if size is not None:
        p.size = size
    return p


class ZerosParam(NewSignalParam):
    """Parameters for zero signal"""

    def generate_1d_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1D data based on current parameters.

        Returns:
            Tuple of (x, y) arrays
        """
        x = self.generate_x_data()
        return x, np.zeros_like(x)


register_signal_parameters_class(SignalTypes.ZEROS, ZerosParam)


class UniformDistribution1DParam(NewSignalParam, base.BaseUniformDistributionParam):
    """Uniform-distribution signal parameters."""

    def generate_1d_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1D data based on current parameters.

        Returns:
            Tuple of (x, y) arrays.
        """
        x = self.generate_x_data()
        rng = np.random.default_rng(self.seed)
        assert self.vmin is not None
        assert self.vmax is not None
        y = self.vmin + rng.random(len(x)) * (self.vmax - self.vmin)
        return x, y


register_signal_parameters_class(
    SignalTypes.UNIFORM_DISTRIBUTION, UniformDistribution1DParam
)


class NormalDistribution1DParam(NewSignalParam, base.BaseNormalDistributionParam):
    """Normal-distribution signal parameters."""

    def generate_1d_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1D data based on current parameters.

        Returns:
            Tuple of (x, y) arrays.
        """
        x = self.generate_x_data()
        rng = np.random.default_rng(self.seed)
        assert self.mu is not None
        assert self.sigma is not None
        y = rng.normal(self.mu, self.sigma, len(x))
        return x, y


register_signal_parameters_class(
    SignalTypes.NORMAL_DISTRIBUTION, NormalDistribution1DParam
)


class PoissonDistribution1DParam(NewSignalParam, base.BasePoissonDistributionParam):
    """Poisson-distribution signal parameters."""

    def generate_1d_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1D data based on current parameters.

        Returns:
            Tuple of (x, y) arrays.
        """
        x = self.generate_x_data()
        rng = np.random.default_rng(self.seed)
        assert self.lam is not None
        y = rng.poisson(lam=self.lam, size=len(x))
        return x, y


register_signal_parameters_class(
    SignalTypes.POISSON_DISTRIBUTION, PoissonDistribution1DParam
)


class BaseGaussLorentzVoigtParam(NewSignalParam):
    """Base parameters for Gaussian, Lorentzian and Voigt functions"""

    STYPE: Type[SignalTypes] | None = None

    a = gds.FloatItem("A", default=1.0)
    y0 = gds.FloatItem("y<sub>0</sub>", default=0.0).set_pos(col=1)
    sigma = gds.FloatItem("σ", default=1.0)
    mu = gds.FloatItem("μ", default=0.0).set_pos(col=1)

    def generate_title(self) -> str:
        """Generate a title based on current parameters."""
        assert isinstance(self.STYPE, SignalTypes)
        return (
            f"{self.STYPE.name.lower()}(a={self.a:.3g},σ={self.sigma:.3g},"
            f"μ={self.mu:.3g},y0={self.y0:.3g})"
        )

    def generate_1d_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1D data based on current parameters.

        Returns:
            Tuple of (x, y) arrays
        """
        x = self.generate_x_data()
        func = {
            SignalTypes.GAUSS: GaussianModel.func,
            SignalTypes.LORENTZ: LorentzianModel.func,
            SignalTypes.VOIGT: VoigtModel.func,
        }[self.STYPE]
        y = func(x, self.a, self.sigma, self.mu, self.y0)
        return x, y


class GaussParam(BaseGaussLorentzVoigtParam):
    """Parameters for Gaussian function"""

    STYPE = SignalTypes.GAUSS


register_signal_parameters_class(SignalTypes.GAUSS, GaussParam)


class LorentzParam(BaseGaussLorentzVoigtParam):
    """Parameters for Lorentzian function"""

    STYPE = SignalTypes.LORENTZ


register_signal_parameters_class(SignalTypes.LORENTZ, LorentzParam)


class VoigtParam(BaseGaussLorentzVoigtParam):
    """Parameters for Voigt function"""

    STYPE = SignalTypes.VOIGT


register_signal_parameters_class(SignalTypes.VOIGT, VoigtParam)


class PlanckParam(NewSignalParam):
    """Planck radiation law.

    y = (2 h c<sup>2</sup>) / (λ<sup>5</sup> (exp(h c / (λ k<sub>B</sub> T)) - 1))
    """

    xmin = gds.FloatItem(
        "λ<sub>min</sub>", default=1e-7, unit="m", min=0.0, nonzero=True
    )
    xmax = gds.FloatItem(
        "λ<sub>max</sub>", default=1e-4, unit="m", min=0.0, nonzero=True
    ).set_prop("display", col=1)
    T = gds.FloatItem(
        "T", default=293.0, unit="K", min=0.0, nonzero=True, help=_("Temperature")
    )

    def generate_title(self) -> str:
        """Generate a title based on current parameters.

        Returns:
            Title string.
        """
        return f"planck(T={self.T:.3g}K)"

    @classmethod
    def func(cls, wavelength: np.ndarray, temperature: float) -> np.ndarray:
        """Compute the Planck function.

        Args:
            wavelength: Wavelength (m).
            T: Temperature (K).

        Returns:
            Spectral radiance (W m<sup>-2</sup> sr<sup>-1</sup> Hz<sup>-1</sup>).
        """
        h = scipy.constants.h  # Planck constant (J·s)
        c = scipy.constants.c  # Speed of light (m/s)
        k = scipy.constants.k  # Boltzmann constant (J/K)
        c1 = 2 * h * c**2
        c2 = (h * c) / k
        denom = np.exp(c2 / (wavelength * temperature)) - 1.0
        spectral_radiance = c1 / (wavelength**5 * (denom))
        return spectral_radiance

    def generate_1d_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1D data based on current parameters.

        Returns:
            Tuple of (wavelength, spectral radiance) arrays.
        """
        wavelength = self.generate_x_data()
        assert self.T is not None
        y = self.func(wavelength, self.T)
        return wavelength, y


register_signal_parameters_class(SignalTypes.PLANCK, PlanckParam)


class FreqUnits(enum.Enum):
    """Frequency units"""

    HZ = "Hz"
    KHZ = "kHz"
    MHZ = "MHz"
    GHZ = "GHz"

    @classmethod
    def convert_in_hz(cls, value, unit):
        """Convert value in Hz"""
        factor = {cls.HZ: 1, cls.KHZ: 1e3, cls.MHZ: 1e6, cls.GHZ: 1e9}.get(unit)
        if factor is None:
            raise ValueError(f"Unknown unit: {unit}")
        return value * factor


class BasePeriodicParam(NewSignalParam):
    """Parameters for periodic functions"""

    STYPE: Type[SignalTypes] | None = None

    def get_frequency_in_hz(self):
        """Return frequency in Hz"""
        return FreqUnits.convert_in_hz(self.freq, self.freq_unit)

    # Redefining some parameters with more appropriate defaults
    xunit = gds.StringItem(_("X unit"), default="s")

    a = gds.FloatItem(_("Amplitude"), default=1.0)
    offset = gds.FloatItem(_("Offset"), default=0.0).set_pos(col=1)
    freq = gds.FloatItem(_("Frequency"), default=1.0)
    freq_unit = gds.ChoiceItem(_("Unit"), FreqUnits, default=FreqUnits.HZ).set_pos(
        col=1
    )
    phase = gds.FloatItem(_("Phase"), default=0.0, unit="°")

    def generate_title(self) -> str:
        """Generate a title based on current parameters."""
        assert isinstance(self.STYPE, SignalTypes)
        freq_hz = self.get_frequency_in_hz()
        title = (
            f"{self.STYPE.name.lower()}(f={freq_hz:.3g}Hz,"
            f"a={self.a:.3g},offset={self.offset:.3g},phase={self.phase:.3g}°)"
        )
        return title

    def generate_1d_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1D data based on current parameters.

        Returns:
            Tuple of (x, y) arrays
        """
        x = self.generate_x_data()
        func = {
            SignalTypes.SINUS: np.sin,
            SignalTypes.COSINUS: np.cos,
            SignalTypes.SAWTOOTH: sps.sawtooth,
            SignalTypes.TRIANGLE: triangle_func,
            SignalTypes.SQUARE: sps.square,
            SignalTypes.SINC: np.sinc,
        }[self.STYPE]
        freq = self.get_frequency_in_hz()
        y = self.a * func(2 * np.pi * freq * x + np.deg2rad(self.phase)) + self.offset
        return x, y


class SinusParam(BasePeriodicParam):
    """Parameters for sinus function"""

    STYPE = SignalTypes.SINUS


register_signal_parameters_class(SignalTypes.SINUS, SinusParam)


class CosinusParam(BasePeriodicParam):
    """Parameters for cosinus function"""

    STYPE = SignalTypes.COSINUS


register_signal_parameters_class(SignalTypes.COSINUS, CosinusParam)


class SawtoothParam(BasePeriodicParam):
    """Parameters for sawtooth function"""

    STYPE = SignalTypes.SAWTOOTH


register_signal_parameters_class(SignalTypes.SAWTOOTH, SawtoothParam)


class TriangleParam(BasePeriodicParam):
    """Parameters for triangle function"""

    STYPE = SignalTypes.TRIANGLE


register_signal_parameters_class(SignalTypes.TRIANGLE, TriangleParam)


class SquareParam(BasePeriodicParam):
    """Parameters for square function"""

    STYPE = SignalTypes.SQUARE


register_signal_parameters_class(SignalTypes.SQUARE, SquareParam)


class SincParam(BasePeriodicParam):
    """Parameters for cardinal sine function"""

    STYPE = SignalTypes.SINC


register_signal_parameters_class(SignalTypes.SINC, SincParam)


class LinearChirpParam(NewSignalParam):
    """Linear chirp function.

    y = y<sub>0</sub> + a sin(φ<sub>0</sub> + 2π (f<sub>0</sub> x + 0.5 k x²))
    """

    a = gds.FloatItem("a", default=1.0, help=_("Amplitude"))
    phi0 = gds.FloatItem(
        "φ<sub>0</sub>", default=0.0, help=_("Initial phase")
    ).set_prop("display", col=1)
    k = gds.FloatItem("k", default=1.0, help=_("Chirp rate (f<sup>-2</sup>)"))
    offset = gds.FloatItem(
        "y<sub>0</sub>", default=0.0, help=_("Vertical offset")
    ).set_prop("display", col=1)
    f0 = gds.FloatItem("f<sub>0</sub>", default=1.0, help=_("Initial frequency (Hz)"))

    def generate_title(self) -> str:
        """Generate a title based on current parameters.

        Returns:
            Title string.
        """
        return (
            f"chirp(a={self.a:.3g},"
            f"k={self.k:.3g},"
            f"f0={self.f0:.3g},"
            f"phi0={self.phi0:.3g},"
            f"ymin={self.offset:.3g})"
        )

    @classmethod
    def func(
        cls, x: np.ndarray, a: float, k: float, f0: float, phi0: float, offset: float
    ) -> np.ndarray:
        """Compute the linear chirp function.

        Args:
            x: X data array.
            a: Amplitude.
            k: Chirp rate (s<sup>-2</sup>).
            f0: Initial frequency (Hz).
            phi0: Initial phase.
            offset: Vertical offset.

        Returns:
            Y data array computed using the chirp function.
        """
        phase = phi0 + 2 * np.pi * (f0 * x + 0.5 * k * x**2)
        return offset + a * np.sin(phase)

    def generate_1d_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1D data based on current parameters.

        Returns:
            Tuple of (x, y) arrays.
        """
        assert self.a is not None
        assert self.k is not None
        assert self.f0 is not None
        assert self.phi0 is not None
        assert self.offset is not None
        x = self.generate_x_data()
        y = self.func(x, self.a, self.k, self.f0, self.phi0, self.offset)
        return x, y


register_signal_parameters_class(SignalTypes.LINEARCHIRP, LinearChirpParam)


class StepParam(NewSignalParam):
    """Parameters for step function"""

    a1 = gds.FloatItem("A<sub>1</sub>", default=0.0)
    a2 = gds.FloatItem("A<sub>2</sub>", default=1.0).set_pos(col=1)
    x0 = gds.FloatItem("x<sub>0</sub>", default=0.0)

    def generate_title(self) -> str:
        """Generate a title based on current parameters."""
        return f"step(a1={self.a1:.3g},a2={self.a2:.3g},x0={self.x0:.3g})"

    def generate_1d_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1D data based on current parameters.

        Returns:
            Tuple of (x, y) arrays
        """
        x = self.generate_x_data()
        y = np.ones_like(x) * self.a1
        y[x > self.x0] = self.a2
        return x, y


register_signal_parameters_class(SignalTypes.STEP, StepParam)


class ExponentialParam(NewSignalParam):
    """Parameters for exponential function"""

    a = gds.FloatItem("A", default=1.0)
    offset = gds.FloatItem(_("Offset"), default=0.0)
    exponent = gds.FloatItem(_("Exponent"), default=1.0)

    def generate_title(self) -> str:
        """Generate a title based on current parameters."""
        return f"exponential(a={self.a:.3g},k={self.exponent:.3g},y0={self.offset:.3g})"

    def generate_1d_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1D data based on current parameters.

        Returns:
            Tuple of (x, y) arrays
        """
        x = self.generate_x_data()
        y = self.a * np.exp(self.exponent * x) + self.offset
        return x, y


register_signal_parameters_class(SignalTypes.EXPONENTIAL, ExponentialParam)


class LogisticParam(NewSignalParam):
    """Logistic function.

    y = y<sub>0</sub> + a / (1 + exp(-k (x - x<sub>0</sub>)))
    """

    a = gds.FloatItem("a", default=1.0, help=_("Amplitude"))
    x0 = gds.FloatItem(
        "x<sub>0</sub>", default=0.0, help=_("Horizontal offset")
    ).set_prop("display", col=1)
    k = gds.FloatItem("k", default=1.0, help=_("Growth or decay rate"))
    offset = gds.FloatItem(
        "y<sub>0</sub>", default=0.0, help=_("Vertical offset")
    ).set_prop("display", col=1)

    def generate_title(self) -> str:
        """Generate a title based on current parameters.

        Returns:
            Title string.
        """
        return (
            f"logistic(a={self.a:.3g},"
            f"k={self.k:.3g},"
            f"x0={self.x0:.3g},"
            f"ymin={self.offset:.3g})"
        )

    @classmethod
    def func(
        cls, x: np.ndarray, a: float, k: float, x0: float, offset: float
    ) -> np.ndarray:
        """Compute the logistic function.

        Args:
            x: X data array.
            a: Amplitude.
            k: Growth or decay rate.
            x0: Horizontal offset.
            offset: Vertical offset.

        Returns:
            Y data array computed using the logistic function.
        """
        return offset + a / (1.0 + np.exp(-k * (x - x0)))

    def generate_1d_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1D data based on current parameters.

        Returns:
            Tuple of (x, y) arrays.
        """
        assert self.a is not None
        assert self.k is not None
        assert self.x0 is not None
        assert self.offset is not None
        x = self.generate_x_data()
        y = self.func(x, self.a, self.k, self.x0, self.offset)
        return x, y


register_signal_parameters_class(SignalTypes.LOGISTIC, LogisticParam)


class PulseParam(NewSignalParam):
    """Parameters for pulse function"""

    amp = gds.FloatItem("Amplitude", default=1.0)
    start = gds.FloatItem(_("Start"), default=0.0).set_pos(col=1)
    offset = gds.FloatItem(_("Offset"), default=0.0)
    stop = gds.FloatItem(_("End"), default=0.0).set_pos(col=1)

    def generate_title(self) -> str:
        """Generate a title based on current parameters."""
        return (
            f"pulse(start={self.start:.3g},stop={self.stop:.3g},"
            f"offset={self.offset:.3g},amp={self.amp:.3g})"
        )

    def generate_1d_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1D data based on current parameters.

        Returns:
            Tuple of (x, y) arrays
        """
        x = self.generate_x_data()
        y = np.full_like(x, self.offset)
        y[(x >= self.start) & (x <= self.stop)] += self.amp
        return x, y


register_signal_parameters_class(SignalTypes.PULSE, PulseParam)


class PolyParam(NewSignalParam):
    """Parameters for polynomial function"""

    a0 = gds.FloatItem("a0", default=1.0)
    a3 = gds.FloatItem("a3", default=0.0).set_pos(col=1)
    a1 = gds.FloatItem("a1", default=1.0)
    a4 = gds.FloatItem("a4", default=0.0).set_pos(col=1)
    a2 = gds.FloatItem("a2", default=0.0)
    a5 = gds.FloatItem("a5", default=0.0).set_pos(col=1)

    def generate_title(self) -> str:
        """Generate a title based on current parameters."""
        return (
            f"polynomial(a0={self.a0:.3g},a1={self.a1:.3g},a2={self.a2:.3g},"
            f"a3={self.a3:.3g},a4={self.a4:.3g},a5={self.a5:.3g})"
        )

    def generate_1d_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1D data based on current parameters.

        Returns:
            Tuple of (x, y) arrays
        """
        x = self.generate_x_data()
        y = np.polyval([self.a5, self.a4, self.a3, self.a2, self.a1, self.a0], x)
        return x, y


register_signal_parameters_class(SignalTypes.POLYNOMIAL, PolyParam)


class CustomSignalParam(NewSignalParam):
    """Parameters for custom signal (e.g. manually defined experimental data)"""

    SIZE_RANGE_ACTIVATION_FLAG = False

    xyarray = gds.FloatArrayItem(
        "XY Values",
        format="%g",
    )

    def setup_array(
        self,
        size: int | None = None,
        xmin: float | None = None,
        xmax: float | None = None,
    ) -> None:
        """Setup the xyarray from size, xmin and xmax (use the current values is not
        provided)

        Args:
            size: xyarray size (default: None)
            xmin: X min (default: None)
            xmax: X max (default: None)
        """
        self.size = size or self.size
        self.xmin = xmin or self.xmin
        self.xmax = xmax or self.xmax
        x_arr = np.linspace(self.xmin, self.xmax, self.size)  # type: ignore
        self.xyarray = np.vstack((x_arr, x_arr)).T

    def generate_title(self) -> str:
        """Generate a title based on current parameters."""
        return f"custom(size={self.size})"

    def generate_1d_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1D data based on current parameters.

        Returns:
            Tuple of (x, y) arrays
        """
        self.setup_array(size=self.size, xmin=self.xmin, xmax=self.xmax)
        x, y = self.xyarray.T
        return x, y


register_signal_parameters_class(SignalTypes.CUSTOM, CustomSignalParam)
check_all_signal_parameters_classes()


def triangle_func(xarr: np.ndarray) -> np.ndarray:
    """Triangle function

    Args:
        xarr: x data
    """
    # ignore warning, as type hint is not handled properly in upstream library
    return sps.sawtooth(xarr, width=0.5)  # type: ignore[no-untyped-def]


SIG_NB = 0


def get_next_signal_number() -> int:
    """Get the next signal number.

    This function is used to keep track of the number of signals created.
    It is typically used to generate unique titles for new signals.

    Returns:
        int: new signal number
    """
    global SIG_NB  # pylint: disable=global-statement
    SIG_NB += 1
    return SIG_NB


def create_signal_from_param(param: NewSignalParam) -> SignalObj:
    """Create a new Signal object from parameters.

    Args:
        param: new signal parameters

    Returns:
        Signal object

    Raises:
        NotImplementedError: if the signal type is not supported
    """
    incr_sig_nb = not param.title
    title = param.title = param.title or DEFAULT_TITLE
    if incr_sig_nb:
        title = f"{title} {get_next_signal_number():d}"
    x, y = param.generate_1d_data()
    gen_title = param.generate_title()
    if gen_title:
        title = gen_title if param.title == DEFAULT_TITLE else param.title
    signal = create_signal(
        title,
        x,
        y,
        units=(param.xunit, param.yunit),
        labels=(param.xlabel, param.ylabel),
    )
    return signal
