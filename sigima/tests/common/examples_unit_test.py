# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Test for documentation examples

This module automatically discovers and executes all Python files in the
doc/examples directory to ensure they run without errors.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

import sigima
from sigima.client.stub import patch_simpleremoteproxy_for_stub

# Check if plotpy is available
try:
    import plotpy  # pylint: disable=unused-import  # noqa: F401

    PLOTPY_AVAILABLE = True
except ImportError:
    PLOTPY_AVAILABLE = False


def get_example_dir() -> Path:
    """Get the path to the examples directory."""
    return Path(sigima.__file__).parent.parent / "doc" / "examples"


def get_example_files() -> list[Path]:
    """Get all Python files in doc/examples directory."""
    examples_dir = get_example_dir()
    if not examples_dir.exists():
        return []

    return [
        f
        for f in examples_dir.glob("*.py")
        if not f.name.startswith("_") and f.name != "__init__.py"
    ]


def test_examples_directory_exists() -> None:
    """Test that the examples directory exists and contains Python files."""
    examples_dir = get_example_dir()
    assert examples_dir.exists(), "doc/examples directory should exist"

    python_files = list(examples_dir.rglob("*.py"))
    assert len(python_files) > 0, "doc/examples should contain at least one Python file"


@pytest.mark.skipif(not PLOTPY_AVAILABLE, reason="PlotPy not installed")
@pytest.mark.parametrize("example_file", get_example_files())
def test_example_execution(example_file: Path) -> None:
    """Test that each example file can be executed without errors."""
    # Special handling for datalab_client.py - needs a stub server
    if example_file.name == "datalab_client.py":
        # Use the utility to patch SimpleRemoteProxy for stub server
        stub_server = patch_simpleremoteproxy_for_stub()
        try:
            # Load and execute the module
            _execute_example_module(example_file)
        finally:
            # Clean up the stub server
            stub_server.stop()
    else:
        # Normal execution for other examples
        _execute_example_module(example_file)


def _execute_example_module(example_file: Path) -> None:
    """Execute an example module file.

    Args:
        example_file: Path to the example Python file to execute
    """
    # Load the module
    spec = importlib.util.spec_from_file_location(
        f"example_{example_file.stem}", str(example_file)
    )
    module = importlib.util.module_from_spec(spec)

    # Add the example directory to sys.path temporarily
    examples_dir = str(example_file.parent)
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)

    try:
        # Execute the module
        spec.loader.exec_module(module)
    finally:
        # Clean up sys.path
        if examples_dir in sys.path:
            sys.path.remove(examples_dir)


if __name__ == "__main__":
    # Run the tests when executed directly
    pytest.main([__file__, "-v"])
