# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Arithmetic parameters unit test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

import pytest

from sigima.params import ArithmeticParam
from sigima.tests import guiutils
from sigima.tests.env import execenv


@pytest.mark.gui
def test_arithmetic_param_interactive():
    """Arithmetic parameters interactive test."""
    with guiutils.lazy_qt_app_context(force=True):
        param = ArithmeticParam()
        if param.edit():
            execenv.print(param)


if __name__ == "__main__":
    test_arithmetic_param_interactive()
