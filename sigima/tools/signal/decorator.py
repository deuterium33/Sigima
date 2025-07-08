# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Decorator for signal functions (see parent package :mod:`sigima.tools.signal`).

"""

from functools import wraps
from typing import Any, Callable

import numpy as np


class SignalCheck:
    """Decorator class to check signal properties."""

    def __init__(
        self,
        x_1d: bool = True,
        x_dtype: type = np.floating,
        x_sorted: bool = True,
        x_evenly_spaced: bool = False,
        y_1d: bool = True,
        y_dtype: type = np.floating,
        x_y_same_size: bool = True,
    ):
        """Initialize the SignalCheck decorator.

        Args:
            x_1d: Check if x is 1-D.
            x_dtype: Expected type of x.
            x_sorted: Check if x is sorted in ascending order.
            x_evenly_spaced: Check if x is evenly spaced.
            y_1d: Check if y is 1-D.
            y_dtype: Expected type of y.
            x_y_same_size: Check if x and y have the same size.
        """
        self.x_1d = x_1d
        self.x_dtype = x_dtype
        self.x_sorted = x_sorted
        self.x_evenly_spaced = x_evenly_spaced

        self.y_1d = y_1d
        self.y_dtype = y_dtype

        self.x_y_same_size = x_y_same_size

    def _check_x(self, x: np.ndarray) -> None:
        """Check x according to the specified conditions.

        Args:
            x: The x input array.

        Raises:
            ValueError: If:
                - x is not 1-D,
                - x is not sorted in ascending order,
                - x is not evenly spaced.
            TypeError: If x is not of the expected type.
        """
        if self.x_1d and x.ndim != 1:
            raise ValueError("x must be 1-D.")
        if not np.issubdtype(x.dtype, self.x_dtype):
            raise TypeError(f"x must be of type {self.x_dtype}, but got {x.dtype}.")
        if self.x_sorted and x.size > 1 and not np.all(np.diff(x) >= 0.0):
            raise ValueError("x must be sorted in ascending order.")
        if self.x_evenly_spaced and x.size > 1:
            diff_x = np.diff(x)
            if not np.allclose(diff_x, np.mean(diff_x)):
                raise ValueError("x must be evenly spaced.")

    def _check_y(self, y: np.ndarray) -> None:
        """Check y according to the specified conditions.

        Args:
            y: The y input array.

        Raises:
            ValueError: If y is not 1-D.
            TypeError: If y is not of the expected type.
        """
        if self.y_1d and y.ndim != 1:
            raise ValueError("y must be 1-D.")
        if not np.issubdtype(y.dtype, self.y_dtype):
            raise TypeError(f"y must be of type {self.y_dtype}, but got {y.dtype}.")

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Callable method to apply the decorator.

        Args:
            func: The function to be decorated.

        Returns:
            The decorated function with signal checks applied.
        """

        @wraps(func)
        def wrapper(x: np.ndarray, y: np.ndarray, *args: Any, **kwargs: Any) -> Any:
            """Wrapper method to apply checks before calling the original function.

            Args:
                x: The x input array.
                y: The y input array.
                *args: Additional positional arguments.
                **kwargs: Additional keyword arguments.

            Returns:
                The result of the original function after checks.
            """
            self._check_x(x)
            self._check_y(y)
            if self.x_y_same_size and x.size != y.size:
                raise ValueError("x and y must have the same size.")
            return func(x, y, *args, **kwargs)

        return wrapper
