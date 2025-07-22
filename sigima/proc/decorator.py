# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Computation function decorator and utilities
(see parent package :mod:`sigima.computation`)
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import inspect
import os.path as osp
import pkgutil
import sys
import typing
from typing import Callable, Literal, TypeVar

import guidata.dataset as gds
import makefun

if sys.version_info >= (3, 10):
    # Use ParamSpec from typing module in Python 3.10+
    from typing import ParamSpec
else:
    # Use ParamSpec from typing_extensions module in Python < 3.10
    from typing_extensions import ParamSpec

__all__ = [
    "computation_function",
    "is_computation_function",
    "get_computation_metadata",
    "find_computation_functions",
]

# Marker attribute used by @computation_function and introspection
COMPUTATION_METADATA_ATTR = "__computation_function_metadata"

P = ParamSpec("P")
R = TypeVar("R")


@dataclasses.dataclass(frozen=True)
class ComputationMetadata:
    """Metadata for a computation function.

    Attributes:
        name: The name of the computation function.
        description: A description or docstring for the computation function.
    """

    name: str
    description: str


def computation_function(
    *,
    name: typing.Optional[str] = None,
    description: typing.Optional[str] = None,
) -> typing.Callable[
    [typing.Callable[..., typing.Any]], typing.Callable[..., typing.Any]
]:
    """
    Decorator to mark a function as a Sigima computation function.

    - Adds a marker attribute (ComputationMetadata) to the wrapped function.
    - Makes the signature explicit: parameters of any guidata DataSet used as argument
      become keyword-only arguments, so the function can be called as
      `func(obj, param1=..., param2=..., ...)`.
    - Maintains backward compatibility: a DataSet instance can still be passed directly.
    - Signature shown in IDE and help() reflects all available parameters.

    Args:
        name: Optional name to override the function name in metadata.
        description: Optional docstring override or additional description.

    Returns:
        The wrapped function, tagged with a marker attribute and an explicit signature.
    """

    def decorator(
        f: typing.Callable[..., typing.Any],
    ) -> typing.Callable[..., typing.Any]:
        metadata = ComputationMetadata(
            name=name or f.__name__, description=description or f.__doc__
        )

        sig = inspect.signature(f)
        params = list(sig.parameters.values())
        try:
            type_hints = typing.get_type_hints(f)
        except Exception:
            type_hints = {}

        ds_param = None
        ds_cls = None
        for p in params:
            annot = type_hints.get(p.name, p.annotation)
            if (
                annot != inspect._empty
                and isinstance(annot, type)
                and issubclass(annot, gds.DataSet)
                and annot.__name__ not in ("SignalObj", "ImageObj")
            ):
                ds_param = p
                ds_cls = annot
                break

        if ds_cls is not None:
            # Build the signature: expose all DataSet items as keyword-only parameters
            items: list[inspect.Parameter] = []
            ds_items: list[gds.DataItem] = ds_cls._items
            for item in ds_items:
                if item.get_name() not in [p.name for p in params]:
                    if isinstance(item, gds.ChoiceItem):
                        choice_data = item.get_prop("data", "choices")
                        choices = [v[0] for v in choice_data]
                        item_type = Literal[tuple(choices)]
                    else:
                        item_type = item.type
                    items.append(
                        inspect.Parameter(
                            item.get_name(),
                            inspect.Parameter.KEYWORD_ONLY,
                            annotation=item_type,
                            default=item.get_default(),
                        )
                    )
            # Keep DataSet param as positional-or-keyword for backward compatibility
            base_params = []
            for p in params:
                if p is ds_param:
                    base_params.append(
                        inspect.Parameter(
                            p.name,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            annotation=p.annotation,
                            default=None,  # Make param optional!
                        )
                    )
                else:
                    base_params.append(p)
            new_params = base_params + items
            new_sig = sig.replace(parameters=new_params)

            # --- inner, makefun-wrapped implementation ---
            @makefun.with_signature(new_sig)
            @functools.wraps(f)
            def real_wrapper(*args, **kwargs):
                user_kwarg_keys = getattr(real_wrapper, "_user_kwarg_keys", set())
                ba = new_sig.bind(*args, **kwargs)
                ba.apply_defaults()
                ds_obj = ba.arguments.get(ds_param.name, None)
                ds_item_names = set([it.get_name() for it in ds_items])
                if isinstance(ds_obj, ds_cls):
                    conflict_keys = ds_item_names.intersection(user_kwarg_keys)
                    if conflict_keys:
                        raise TypeError(
                            f"Cannot pass both a {ds_cls.__name__} instance and "
                            f"keyword arguments for its items "
                            f"({', '.join(conflict_keys)}). Please use only one style."
                        )
                else:
                    # DataSet instance NOT provided: build from keyword-only arguments
                    ds_kwargs = {
                        k: ba.arguments.pop(k)
                        for k in list(ba.arguments.keys())
                        if k in ds_item_names
                    }
                    ds_obj = ds_cls.create(**ds_kwargs)
                final_args = []
                for p in params:
                    if p is ds_param:
                        final_args.append(ds_obj)
                    else:
                        final_args.append(ba.arguments.get(p.name, None))
                return f(*final_args)

            # --- outer pre-wrapper: captures user-passed keywords only ---
            def pre_wrapper(*args, **kwargs):
                real_wrapper._user_kwarg_keys = set(kwargs.keys())
                return real_wrapper(*args, **kwargs)

            # --- preserve the original function's docstring ---
            pre_wrapper.__doc__ = f.__doc__
            pre_wrapper.__name__ = f.__name__

            # --- Sphinx-style docstring injection with actual parameter names ---
            param_class_name = ds_cls.__name__
            item_names = [item.get_name() for item in ds_items]
            kwarg_example = ", ".join(f"{name}=..." for name in item_names)

            signature_info = (
                f".. note::\n\n"
                f"   This computation function can be called in two ways:\n\n"
                f"   1. With a parameter ``{param_class_name}`` object:\n\n"
                f"   .. code-block:: python\n\n"
                f"       param = {param_class_name}.create({kwarg_example})\n"
                f"       func(obj, param)\n\n"
                f"   2. Or, with keyword arguments directly:\n\n"
                f"   .. code-block:: python\n\n"
                f"       func(obj, {kwarg_example})\n\n"
            )
            doc = f.__doc__ or ""
            if not doc.endswith("\n"):
                doc += "\n"
            pre_wrapper.__doc__ = doc + signature_info

            setattr(pre_wrapper, COMPUTATION_METADATA_ATTR, metadata)
            pre_wrapper.__signature__ = new_sig
            return pre_wrapper

        else:

            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs)

            setattr(wrapper, COMPUTATION_METADATA_ATTR, metadata)
            return wrapper

    return decorator


def is_computation_function(function: Callable) -> bool:
    """Check if a function is a Sigima computation function.

    Args:
        function: The function to check.

    Returns:
        True if the function is a Sigima computation function, False otherwise.
    """
    return getattr(function, COMPUTATION_METADATA_ATTR, None) is not None


def get_computation_metadata(function: Callable) -> ComputationMetadata:
    """Get the metadata of a Sigima computation function.

    Args:
        function: The function to get metadata from.

    Returns:
        Computation function metadata.

    Raises:
        ValueError: If the function is not a Sigima computation function.
    """
    metadata = getattr(function, COMPUTATION_METADATA_ATTR, None)
    if not isinstance(metadata, ComputationMetadata):
        raise ValueError(
            f"The function {function.__name__} is not a Sigima computation function."
        )
    return metadata


def find_computation_functions() -> list[tuple[str, Callable]]:
    """Find all computation functions in the `sigima.proc` package.

    This function uses introspection to locate all functions decorated with
    `@computation_function` in the `sigima.proc` package and its subpackages.

    Args:
        module: Optional module to search in. If None, the current module is used.

    Returns:
        A list of tuples, each containing the function name and the function object.
    """
    functions = []
    objs = []
    for _, modname, _ in pkgutil.walk_packages(
        path=[osp.dirname(__file__)], prefix=".".join(__name__.split(".")[:-1]) + "."
    ):
        try:
            module = importlib.import_module(modname)
        except Exception:  # pylint: disable=broad-except
            continue
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if is_computation_function(obj):
                if obj in objs:  # Avoid double entries for the same function
                    continue
                objs.append(obj)
                functions.append((modname, name, obj.__doc__))
    return functions
