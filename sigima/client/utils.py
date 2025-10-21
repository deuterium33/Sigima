# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Sigima Client Utilities
-----------------------

Common utilities shared between client remote control and stub server modules.
"""

from __future__ import annotations

import importlib
from io import BytesIO
from xmlrpc.client import Binary

import guidata.dataset as gds
import numpy as np
from guidata.io import JSONReader, JSONWriter

# pylint: disable=duplicate-code


def array_to_rpcbinary(data: np.ndarray) -> Binary:
    """Convert NumPy array to XML-RPC Binary object, with shape and dtype.

    The array is converted to a binary string using NumPy's native binary
    format.

    Args:
        data: NumPy array to convert

    Returns:
        XML-RPC Binary object
    """
    dbytes = BytesIO()
    np.save(dbytes, data, allow_pickle=False)
    return Binary(dbytes.getvalue())


def rpcbinary_to_array(binary: Binary) -> np.ndarray:
    """Convert XML-RPC binary to NumPy array.

    Args:
        binary: XML-RPC Binary object

    Returns:
        NumPy array
    """
    dbytes = BytesIO(binary.data)
    return np.load(dbytes, allow_pickle=False)


def dataset_to_rpcjson(param: gds.DataSet) -> list[str]:
    """Convert guidata DataSet to XML-RPC compatible JSON data.

    The JSON data is a list of three elements:

    - The first element is the module name of the DataSet class
    - The second element is the class name of the DataSet class
    - The third element is the JSON data of the DataSet instance

    Args:
        param: guidata DataSet to convert

    Returns:
        XML-RPC compatible JSON data (3-element list)
    """
    writer = JSONWriter()
    param.serialize(writer)
    param_json = writer.get_json()
    klass = param.__class__
    return [klass.__module__, klass.__name__, param_json]


def rpcjson_to_dataset(param_data: list[str]) -> gds.DataSet:
    """Convert XML-RPC compatible JSON data to guidata DataSet.

    Args:
        param_data: XML-RPC compatible JSON data (3-element list)

    Returns:
        guidata DataSet
    """
    param_module, param_clsname, param_json = param_data
    mod = importlib.__import__(param_module, fromlist=[param_clsname])
    klass = getattr(mod, param_clsname)
    param: gds.DataSet = klass()
    reader = JSONReader(param_json)
    param.deserialize(reader)
    return param
