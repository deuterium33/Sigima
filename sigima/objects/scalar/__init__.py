# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Scalar results subpackage
=========================

This subpackage provides classes and functions for handling scalar results in Sigima.

The subpackage is split into two main modules:

- :mod:`sigima.objects.scalar.table`: Table results and related utilities
- :mod:`sigima.objects.scalar.geometry`: Geometry results and related utilities

For backward compatibility, all public symbols are re-exported from this __init__.py
file, so existing imports like:

.. code-block:: python

    from sigima.objects.scalar import TableResult, GeometryResult

continue to work as expected.
"""

# Import all public symbols from both modules
from sigima.objects.scalar.geometry import (
    GeometryResult,
    KindShape,
    concat_geometries,
    filter_geometry_by_roi,
)
from sigima.objects.scalar.table import (
    NO_ROI,
    ResultHtmlGenerator,
    TableKind,
    TableResult,
    TableResultBuilder,
    calc_table_from_data,
    concat_tables,
    filter_table_by_roi,
)

# Define __all__ to specify what gets imported with
# "from sigima.objects.scalar import *"
__all__ = [
    "NO_ROI",
    "GeometryResult",
    "KindShape",
    "ResultHtmlGenerator",
    "TableKind",
    "TableResult",
    "TableResultBuilder",
    "calc_table_from_data",
    "concat_geometries",
    "concat_tables",
    "filter_geometry_by_roi",
    "filter_table_by_roi",
]
