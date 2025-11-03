"""Test to verify func_name is systematically set by @computation_function decorator"""
# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

import sigima.objects
import sigima.params
import sigima.proc.signal
import sigima.tests.data


def test_func_name_auto_injection():
    """Verify that @computation_function decorator automatically sets func_name."""
    # Create a test signal
    obj = sigima.tests.data.get_test_signal("fwhm.txt")

    # Test 1: Function with parameters (fwhm)
    param = sigima.params.FWHMParam.create(method="gauss")
    result = sigima.proc.signal.fwhm(obj, param)
    assert result is not None
    assert hasattr(result, "func_name")
    assert result.func_name == "fwhm", f"Expected 'fwhm', got {result.func_name!r}"

    # Test 2: Function without parameters (fw1e2)
    result2 = sigima.proc.signal.fw1e2(obj)
    assert result2 is not None
    assert hasattr(result2, "func_name")
    assert result2.func_name == "fw1e2", f"Expected 'fw1e2', got {result2.func_name!r}"

    print("✓ All func_name auto-injection tests passed!")


if __name__ == "__main__":
    test_func_name_auto_injection()
