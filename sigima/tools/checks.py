# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Checks for 1D and 2D NumPy arrays used in tools (:mod:`sigima.tools.checks`).
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import numpy as np


def check_1d_arrays(
    func: Callable[..., Any] | None = None,
    *,
    x_1d: bool = True,
    x_dtype: type = np.floating,
    x_sorted: bool = False,
    x_evenly_spaced: bool = False,
    y_1d: bool = True,
    y_dtype: type = np.floating,
    x_y_same_size: bool = True,
    rtol: float = 1e-5,
) -> Callable:
    """
    Decorator to check inputs of functions operating on 1D NumPy arrays (x/y).

    Can be used with parentheses:

    .. code-block:: python

        @check_1d_arrays(x_1d=True, y_1d=True)
        def process_signals(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            # Process the signals
            return x + y

    Or without parentheses (default arguments):

    .. code-block:: python

        @check_1d_arrays
        def process_signals(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            # Process the signals
            return x + y

    Args:
        x_1d: Whether to check if x is 1-D.
        x_dtype: Expected dtype of x.
        x_sorted: Whether to check if x is sorted.
        x_evenly_spaced: Whether to check if x is evenly spaced.
        y_1d: Whether to check if y is 1-D.
        y_dtype: Expected dtype of y.
        x_y_same_size: Whether to check if x and y have same size.
        rtol: Relative tolerance for regular spacing.

    Returns:
        Decorated function with pre-checks on x/y.
    """

    def decorator(inner_func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(inner_func)
        def wrapper(x: np.ndarray, y: np.ndarray, *args: Any, **kwargs: Any) -> Any:
            # === Check x array
            if x_1d and x.ndim != 1:
                raise ValueError("x must be 1-D.")
            if not np.issubdtype(x.dtype, x_dtype):
                raise TypeError(f"x must be of type {x_dtype}, but got {x.dtype}.")
            if x_sorted and x.size > 1 and not np.all(np.diff(x) >= 0.0):
                raise ValueError("x must be sorted in ascending order.")
            if x_evenly_spaced and x.size > 1:
                dx = np.diff(x)
                if not np.allclose(dx, np.mean(dx), rtol=rtol):
                    raise ValueError("x must be evenly spaced.")
            # === Check y array
            if y_1d and y.ndim != 1:
                raise ValueError("y must be 1-D.")
            if not np.issubdtype(y.dtype, y_dtype):
                raise TypeError(f"y must be of type {y_dtype}, but got {y.dtype}.")
            if x_y_same_size and x.size != y.size:
                raise ValueError("x and y must have the same size.")
            # === Call the original function
            return inner_func(x, y, *args, **kwargs)

        return wrapper

    if func is not None:
        # Usage: `@check_1d_arrays`
        return decorator(func)
    # Usage: `@check_1d_arrays(...)`
    return decorator


def check_2d_array(
    func: Callable[..., Any] | None = None,
    *,
    ndim: int = 2,
    dtype: type | None = None,
    non_constant: bool = False,
    finite_only: bool = False,
) -> Callable:
    """
    Decorator to check input for functions operating on 2D NumPy arrays (e.g. images).

    Can be used with parentheses:

    .. code-block:: python

        @check_2d_array(ndim=3, dtype=np.uint8)
        def process_image(image: np.ndarray) -> np.ndarray:
            # Process the image
            return image

    Or without parentheses (default arguments):

    .. code-block:: python

        @check_2d_array
        def process_image(image: np.ndarray) -> np.ndarray:
            # Process the image
            return image

    Args:
        ndim: Expected number of dimensions.
        dtype: Expected dtype.
        non_constant: Whether to check that the array has dynamic range.
        finite_only: Whether to check that all values are finite.

    Returns:
        Decorated function with pre-checks on data.
    """

    def decorator(inner_func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(inner_func)
        def wrapper(data: np.ndarray, *args: Any, **kwargs: Any) -> Any:
            # === Check input array
            if data.ndim != ndim:
                raise ValueError(f"Input array must be {ndim}D, got {data.ndim}D.")
            if dtype is not None and not np.issubdtype(data.dtype, dtype):
                raise TypeError(
                    f"Input array must be of type {dtype}, got {data.dtype}."
                )
            if non_constant:
                dmin, dmax = np.nanmin(data), np.nanmax(data)
                if dmin == dmax:
                    raise ValueError("Input array has no dynamic range.")
            if finite_only and not np.all(np.isfinite(data)):
                raise ValueError("Input array contains non-finite values.")
            # === Call the original function
            return inner_func(data, *args, **kwargs)

        return wrapper

    if func is not None:
        # Usage: `@check_2d_array`
        return decorator(func)
    # Usage: `@check_2d_array(...)`
    return decorator
