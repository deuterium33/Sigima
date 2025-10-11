# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
I/O test

Testing `sigima` specific formats.
"""

from __future__ import annotations

import os.path as osp

from sigima.io.image import ImageIORegistry
from sigima.io.signal import SignalIORegistry
from sigima.tests.env import execenv
from sigima.tests.helpers import WorkdirRestoringTempDir, read_test_objects, reduce_path


def __testfunc(
    title: str,
    registry: SignalIORegistry | ImageIORegistry,
    pattern: str = "*.*",
    in_folder: str | None = None,
) -> None:
    """Test I/O features: read and write objects, and check registry functionality

    Args:
        title: Title of the test
        registry: I/O registry to use
        pattern: File name pattern to match
        in_folder: Folder to search for test files

    Raises:
        NotImplementedError: if format is not supported
    """
    execenv.print(f"  {title}:")
    with WorkdirRestoringTempDir() as tmpdir:
        # os.startfile(tmpdir)
        objects = {}
        for fname, obj in read_test_objects(registry, pattern, in_folder):
            label = f"    Opening {reduce_path(fname)}"
            execenv.print(label + ": ", end="")
            if obj is None:
                execenv.print("Skipped (not implemented)")
            else:
                execenv.print("OK")
                objects[fname] = obj
        execenv.print("    Saving:")
        for fname, obj in objects.items():
            path = osp.join(tmpdir, osp.basename(fname))
            try:
                execenv.print(f"      {path}: ", end="")
                registry.write(path, obj)
                execenv.print("OK")
            except NotImplementedError:
                execenv.print("Skipped (not implemented)")


def test_read_write_obj():
    """I/O test: read and write objects, check registry functionality"""
    execenv.print("I/O unit test:")
    __testfunc("Signals", SignalIORegistry)
    __testfunc("Images", ImageIORegistry)


if __name__ == "__main__":
    test_read_write_obj()
