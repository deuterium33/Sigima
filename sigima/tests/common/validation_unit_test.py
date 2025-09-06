# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Testing validation test introspection and CSV generation.
"""

import sigima.tests as tests_pkg
from sigima.proc.decorator import find_computation_functions
from sigima.proc.validation import (
    ValidationStatistics,
    generate_valid_test_names_for_function,
    get_validation_tests,
)
from sigima.tests.helpers import WorkdirRestoringTempDir


def __generate_all_valid_test_names() -> set[str]:
    """Generate all valid test names for all computation functions.

    Returns:
        Set of all valid test names that could test computation functions
    """
    computation_functions = find_computation_functions()
    valid_test_names = set()

    for module_name, func_name, _ in computation_functions:
        names = generate_valid_test_names_for_function(module_name, func_name)
        valid_test_names.update(names)

    return valid_test_names


def test_validation_statistics() -> None:
    """Test validation statistics introspection and CSV generation."""
    stats = ValidationStatistics()
    stats.collect_validation_status(verbose=True)
    stats.get_validation_info()
    with WorkdirRestoringTempDir() as tmpdir:
        stats.generate_csv_files(tmpdir)
        stats.generate_statistics_csv(tmpdir)


def test_validation_decorator_only_on_computation_functions() -> None:
    """Test that @pytest.mark.validation is only used on computation function tests.

    This test ensures that validation tests marked with @pytest.mark.validation
    are only used for testing actual computation functions (those decorated with
    @computation_function). Test functions for non-computation functions (like
    I/O convenience functions) should not have this decorator.
    """
    # Get all functions marked with @pytest.mark.validation
    validation_tests = get_validation_tests(tests_pkg)

    # Get all valid test names for computation functions
    valid_test_names = __generate_all_valid_test_names()

    # Check each validation test to see if it corresponds to a computation function
    invalid_validation_tests = []

    for test_name, test_path, line_number in validation_tests:
        if test_name not in valid_test_names:
            # This validation test doesn't correspond to any computation function
            import os.path as osp

            rel_path = osp.relpath(test_path, start=osp.dirname(tests_pkg.__file__))
            module_parts = rel_path.replace(osp.sep, ".").replace(".py", "")
            module_name = f"sigima.tests.{module_parts}"

            invalid_validation_tests.append((test_name, module_name, line_number))

    # Report any invalid validation tests
    if invalid_validation_tests:
        error_messages = []
        error_messages.append(
            "Found @pytest.mark.validation decorator on tests that don't test "
            "computation functions:"
        )
        for test_name, module_name, line_number in invalid_validation_tests:
            # Convert module path back to file path for clickable links
            file_path = (
                module_name.replace("sigima.tests.", "").replace(".", "\\") + ".py"
            )
            error_messages.append(f"  - {file_path}:{line_number} ({test_name})")
        error_messages.append("")
        error_messages.append(f"Found {len(invalid_validation_tests)} invalid cases.")
        error_messages.append(
            "The @pytest.mark.validation decorator should only be used on "
            "test functions that test computation functions (those decorated with "
            "@computation_function). Please remove this decorator from test functions "
            "that test non-computation functions."
        )

        raise AssertionError("\n".join(error_messages))


if __name__ == "__main__":
    test_validation_statistics()
    test_validation_decorator_only_on_computation_functions()
