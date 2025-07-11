# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Decorator for signal functions (see parent package :mod:`sigima.tools.signal`).

"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import numpy as np


def signal_check(
    func: Callable[..., Any] | None = None,
    *,
    x_1d: bool = True,
    x_dtype: type = np.floating,
    x_sorted: bool = True,
    x_evenly_spaced: bool = False,
    y_1d: bool = True,
    y_dtype: type = np.floating,
    x_y_same_size: bool = True,
) -> Callable:
    """Decorator function to check signal properties.

    Args:
        func: The function to be decorated.
        x_1d: Check if x is 1-D.
        x_dtype: Expected type of x.
        x_sorted: Check if x is sorted in ascending order.
        x_evenly_spaced: Check if x is evenly spaced.
        y_1d: Check if y is 1-D.
        y_dtype: Expected type of y.
        x_y_same_size: Check if x and y have the same size.

    Returns:
        A decorator that applies the specified signal checks.
    """

    def decorator(inner_func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator function to apply signal checks.

        Args:
            inner_func: The function to be decorated.

        Returns:
            The decorated function with signal checks applied.
        """

        @wraps(inner_func)
        def wrapper(x: np.ndarray, y: np.ndarray, *args: Any, **kwargs: Any) -> Any:
            """Wrapper method to apply checks before calling the original function.

            Args:
                x: The x input array.
                y: The y input array.
                *args: Additional positional arguments.
                **kwargs: Additional keyword arguments.

            Returns:
                The result of the original function after checks.

            Raises:
                ValueError: If any of the following conditions is not met:
                    - x must be 1-D if x_1d is True.
                    - x must be sorted in ascending order if x_sorted is True.
                    - x must be evenly spaced if x_evenly_spaced is True.
                    - y must be 1-D if y_1d is True.
                    - x and y must have the same size if x_y_same_size is True.
                TypeError: If any of the following conditions is not met:
                    - x must be of type x_dtype.
                    - y must be of type y_dtype.
            """
            # === Check x array
            if x_1d and x.ndim != 1:
                raise ValueError("x must be 1-D.")
            if not np.issubdtype(x.dtype, x_dtype):
                raise TypeError(f"x must be of type {x_dtype}, but got {x.dtype}.")
            if x_sorted and x.size > 1 and not np.all(np.diff(x) >= 0.0):
                raise ValueError("x must be sorted in ascending order.")
            if x_evenly_spaced and x.size > 1:
                diff_x = np.diff(x)
                if not np.allclose(diff_x, np.mean(diff_x)):
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
        # Usage: `@signal_check`
        return decorator(func)

    # Usage: `@signal_check(...)`
    return decorator
