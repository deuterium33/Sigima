# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Base signal processing functions and utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from sigima.objects import NO_ROI, GeometryResult, KindShape, SignalObj
from sigima.proc.base import dst_1_to_1


def restore_data_outside_roi(dst: SignalObj, src: SignalObj) -> None:
    """Restore data outside the Region Of Interest (ROI) of the input signal
    after a computation, only if the input signal has a ROI,
    and if the output signal has the same ROI as the input signal,
    and if the data types are the same,
    and if the shapes are the same.
    Otherwise, do nothing.

    Args:
        dst: destination signal object
        src: source signal object
    """
    if src.maskdata is not None and dst.maskdata is not None:
        if (
            np.array_equal(src.maskdata, dst.maskdata)
            and dst.xydata.dtype == src.xydata.dtype
            and dst.xydata.shape == src.xydata.shape
        ):
            dst.xydata[src.maskdata] = src.xydata[src.maskdata]


def is_uncertainty_data_available(signals: SignalObj | list[SignalObj]) -> bool:
    """Check if all signals have uncertainty data.

    This functions is used to determine whether enough information is available to
    propagate uncertainty.

    Args:
        signals: Signal object or list of signal objects.

    Returns:
        True if all signals have uncertainty data, False otherwise.
    """
    if isinstance(signals, SignalObj):
        signals = [signals]
    return all(sig.dy is not None for sig in signals)


class Wrap1to1Func:
    """Wrap a 1 array → 1 array function (the simple case of y1 = f(y0)) to produce
    a 1 signal → 1 signal function, which can be used as a Sigima computation function
    and inside DataLab's infrastructure to perform computations with the Signal
    Processor object.

    This wrapping mechanism using a class is necessary for the resulted function to be
    pickable by the ``multiprocessing`` module.

    The instance of this wrapper is callable and returns
    a :class:`sigima.objects.SignalObj` object.

    Example:

        >>> import numpy as np
        >>> from sigima.proc.signal import Wrap1to1Func
        >>> import sigima.objects
        >>> def square(y):
        ...     return y**2
        >>> compute_square = Wrap1to1Func(square)
        >>> x = np.linspace(0, 10, 100)
        >>> y = np.sin(x)
        >>> sig0 = sigima.objects.create_signal("Example", x, y)
        >>> sig1 = compute_square(sig0)

    Args:
        func: 1 array → 1 array function
        *args: Additional positional arguments to pass to the function
        **kwargs: Additional keyword arguments to pass to the function

    .. note::

        If `func_name` is provided in the keyword arguments, it will be used as the
        function name instead of the default name derived from the function itself.

    .. note::

        This wrapper is suitable for functions that don't require custom uncertainty
        propagation. For mathematical functions with specific uncertainty formulas
        (sqrt, log10, exp, etc.), implement uncertainty propagation directly in the
        computation function.
    """

    def __init__(self, func: Callable, *args: Any, **kwargs: Any) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.__name__ = self.kwargs.pop("func_name", func.__name__)
        self.__doc__ = func.__doc__
        self.__call__.__func__.__doc__ = self.func.__doc__

    def __call__(self, src: SignalObj) -> SignalObj:
        """Compute the function on the input signal and return the result signal

        Args:
            src: input signal object

        Returns:
            Result signal object
        """
        suffix = ", ".join(
            [str(arg) for arg in self.args]
            + [f"{k}={v}" for k, v in self.kwargs.items() if v is not None]
        )
        dst = dst_1_to_1(src, self.__name__, suffix)
        x, y = src.get_data()
        # Apply function and propagate uncertainty unchanged
        dst.set_xydata(x, self.func(y, *self.args, **self.kwargs), src.dx, src.dy)

        restore_data_outside_roi(dst, src)
        return dst


def compute_geometry_from_obj(
    title: str,
    shape: str | KindShape,
    obj: SignalObj,
    func: Callable,
    *args: Any,
) -> GeometryResult | None:
    """Calculate result geometry by executing a computation function on a signal object.

    Args:
        title: Result title
        shape: Result shape kind (e.g., "segment", "point", KindShape.MARKER)
        obj: Input signal object
        func: Computation function that takes (x, y, ``*args``) and returns coordinates
        *args: Additional computation function arguments

    Returns:
        Result geometry object or None if no result is found

    .. note::

        The computation function must take x and y arrays as the first two arguments,
        followed by any additional arguments, and return a NumPy array containing
        coordinate pairs in the form ``[[x0, y0], [x1, y1], ...]``.
    """
    rows: list[np.ndarray] = []
    roi_idx: list[int] = []

    for i_roi in obj.iterate_roi_indices():
        x, y = obj.get_data(i_roi)
        if args:
            results: np.ndarray = func(x, y, *args)
        else:
            results: np.ndarray = func(x, y)

        if results is None:
            continue

        results = np.array(results, dtype=float)
        if results.size == 0:
            continue

        # Ensure results are in the correct 2D format
        if results.ndim == 1:
            # For segment shapes, expect 4 coordinates: [x0, y0, x1, y1]
            if shape in ("segment", KindShape.SEGMENT) and len(results) == 4:
                results = results.reshape(1, 4)
            elif len(results) % 2 == 0:
                # Reshape flat coordinate array to pairs for points/markers
                results = results.reshape(-1, 2)
            else:
                continue  # Skip malformed results
        elif results.ndim != 2 or results.shape[1] < 2:
            continue  # Skip malformed results

        rows.append(results)
        roi_idx.extend([NO_ROI if i_roi is None else int(i_roi)] * results.shape[0])

    if not rows:
        return None

    coords = np.vstack(rows)

    # Convert shape to KindShape enum
    if isinstance(shape, KindShape):
        shape_kind = shape
    elif shape == "segment":
        shape_kind = KindShape.SEGMENT
    elif shape == "point":
        shape_kind = KindShape.POINT
    elif shape == "marker":
        shape_kind = KindShape.MARKER
    else:
        shape_kind = KindShape.POINT  # Default fallback

    return GeometryResult(
        title=title,
        kind=shape_kind,
        coords=coords,
        roi_indices=np.array(roi_idx, dtype=int),
    )
