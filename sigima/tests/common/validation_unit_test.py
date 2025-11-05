# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Testing validation test introspection and CSV generation.
"""

import inspect
import os.path as osp
import re

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


def test_validation_missing_tests() -> None:
    """Test that all computation functions have validation tests.

    This test ensures that all computation functions (those decorated with
    @computation_function) have corresponding validation tests
    marked with @pytest.mark.validation.
    """
    # Get all functions marked with @pytest.mark.validation
    validation_tests = get_validation_tests(tests_pkg)

    # Get all computation functions that should have validation tests
    computation_functions = find_computation_functions()
    required_functions = [
        (module_name, func_name) for module_name, func_name, _ in computation_functions
    ]

    # Check each required function to see if it has a corresponding validation test
    missing_validation_tests = []

    for module_name, func_name in required_functions:
        valid_test_names = generate_valid_test_names_for_function(
            module_name, func_name
        )

        # Check if any of the valid test names exist in validation tests
        has_validation_test = any(
            test_name in [vt[0] for vt in validation_tests]
            for test_name in valid_test_names
        )

        if not has_validation_test:
            missing_validation_tests.append(f"{module_name}.{func_name}")

    # Report any missing validation tests
    if missing_validation_tests:
        error_messages = []
        error_messages.append(
            "The following computation functions are missing "
            "validation tests marked with @pytest.mark.validation:"
        )
        for func_name in missing_validation_tests:
            error_messages.append(f"  - {func_name}")
        error_messages.append("")
        error_messages.append(f"Found {len(missing_validation_tests)} missing cases.")
        error_messages.append(
            "Please add validation tests for these computation functions."
        )

        raise AssertionError("\n".join(error_messages))


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


def test_computation_functions_documented_in_features() -> None:
    """Test that all computation functions are documented in doc/features.rst.

    This test ensures that all computation functions (those decorated with
    @computation_function) are documented in the features.rst file with
    proper Sphinx :func: references using simplified paths like
    sigima.proc.image.function_name or sigima.proc.signal.function_name.
    """
    # Read the features.rst file
    doc_dir = osp.join(osp.dirname(tests_pkg.__file__), "..", "..", "doc")
    features_rst_path = osp.join(doc_dir, "user_guide", "features.rst")

    if not osp.exists(features_rst_path):
        raise AssertionError(f"Documentation file not found: {features_rst_path}")

    with open(features_rst_path, encoding="utf-8") as f:
        features_content = f.read()

    # Get all computation functions
    computation_functions = find_computation_functions()

    # Check each computation function to see if it's documented
    missing_documentation = []

    for module_name, func_name, _ in computation_functions:
        # Build the expected documentation reference using simplified path
        # The module_name is like "sigima.proc.image" or "sigima.proc.signal"
        module_path = module_name.split("sigima.proc.")[-1]
        expected_ref = f"sigima.proc.{module_path}.{func_name}"

        # Check if this reference exists in the documentation
        if expected_ref not in features_content:
            missing_documentation.append((module_name, func_name, expected_ref))

    # Report any missing documentation
    if missing_documentation:
        error_messages = []
        error_messages.append(
            "The following computation functions are missing from doc/features.rst:"
        )
        for module_name, func_name, expected_ref in missing_documentation:
            error_messages.append(f"  - {func_name} ({expected_ref})")
        error_messages.append("")
        error_messages.append(f"Found {len(missing_documentation)} missing cases.")
        error_messages.append(
            "Please add documentation references for these computation functions "
            "in doc/features.rst using the format:"
        )
        error_messages.append(
            "   :func:`function_name <sigima.proc.module.function_name>`"
        )

        raise AssertionError("\n".join(error_messages))


def _check_function_referenced_in_code(
    source_code: str, module_name: str, func_name: str
) -> bool:
    """Check if a computation function is referenced in source code.

    This is a simple check that looks for the fully qualified function name
    anywhere in the code (e.g., "sigima.proc.image.fliph").

    Args:
        source_code: Source code to check
        module_name: Module name (e.g., "sigima.proc.image")
        func_name: Function name (e.g., "fliph")

    Returns:
        True if the function is referenced in the code
    """
    # Simply check if the full qualified name appears in the source
    full_name = f"{module_name}.{func_name}"
    return full_name in source_code


def _get_helper_functions_called(source_code: str) -> set[str]:
    """Extract names of helper functions called in source code.

    Args:
        source_code: Source code to analyze

    Returns:
        Set of function names that are called
    """
    # Find all function calls that start with __ or _ (helper functions)
    pattern = r"\b(__\w+|_\w+)\s*\("
    matches = re.findall(pattern, source_code)
    return set(matches)


def test_validation_tests_call_computation_functions() -> None:
    """Test that validation tests actually call the computation functions they test.

    This test ensures that each validation test marked with @pytest.mark.validation
    references the corresponding computation function (those decorated with
    @computation_function) either directly or through helper functions in the
    same test module.
    """
    # Get all functions marked with @pytest.mark.validation
    validation_tests = get_validation_tests(tests_pkg)

    # Get all computation functions
    computation_functions = find_computation_functions()

    # Build a mapping from test names to computation functions
    test_to_function_map = {}
    for module_name, func_name, _ in computation_functions:
        valid_test_names = generate_valid_test_names_for_function(
            module_name, func_name
        )
        for test_name in valid_test_names:
            if test_name not in test_to_function_map:
                test_to_function_map[test_name] = []
            test_to_function_map[test_name].append((module_name, func_name))

    # Check each validation test to ensure it references the computation function
    tests_not_calling_function = []

    # pylint: disable=too-many-nested-blocks
    for test_name, test_path, line_number in validation_tests:
        if test_name not in test_to_function_map:
            # This test doesn't correspond to any computation function
            # (will be caught by another test)
            continue

        # Get the source code of the test function
        # Import the test module and get the function
        rel_path = osp.relpath(test_path, start=osp.dirname(tests_pkg.__file__))
        module_parts = rel_path.replace(osp.sep, ".").replace(".py", "")
        test_module_name = f"sigima.tests.{module_parts}"

        try:
            test_module = __import__(test_module_name, fromlist=[test_name])
            test_func = getattr(test_module, test_name)
            test_source_code = inspect.getsource(test_func)
        except (ImportError, AttributeError, OSError):
            # Skip if we can't get the source (e.g., built-in functions)
            continue

        # Check if the computation function is referenced in the test
        for module_name, func_name in test_to_function_map[test_name]:
            # Check if it's referenced directly in the test
            function_referenced = _check_function_referenced_in_code(
                test_source_code, module_name, func_name
            )

            # If not found directly, check helper functions
            if not function_referenced:
                helper_funcs = _get_helper_functions_called(test_source_code)

                for helper_name in helper_funcs:
                    try:
                        helper_func = getattr(test_module, helper_name, None)
                        if helper_func and callable(helper_func):
                            helper_source = inspect.getsource(helper_func)
                            if _check_function_referenced_in_code(
                                helper_source, module_name, func_name
                            ):
                                function_referenced = True
                                break
                    except (AttributeError, OSError):
                        # Can't get source for this helper, skip it
                        continue

            if not function_referenced:
                tests_not_calling_function.append(
                    (test_name, module_name, func_name, test_path, line_number)
                )

    # Report any validation tests that don't call their computation functions
    if tests_not_calling_function:
        error_messages = []
        error_messages.append(
            "The following validation tests don't call their corresponding "
            "computation functions:"
        )
        for (
            test_name,
            module_name,
            func_name,
            test_path,
            line_number,
        ) in tests_not_calling_function:
            rel_path = osp.relpath(test_path, start=osp.dirname(tests_pkg.__file__))
            file_path = rel_path.replace(osp.sep, "\\")
            error_messages.append(
                f"  - {file_path}:{line_number} ({test_name})\n"
                f"    Expected to call: {module_name}.{func_name}"
            )
        error_messages.append("")
        error_messages.append(f"Found {len(tests_not_calling_function)} invalid cases.")
        error_messages.append(
            "Validation tests must call the computation function they are testing, "
            "not the underlying utility functions. Please update these tests to call "
            "the computation function (e.g., sigima.proc.image.func_name(...)) "
            "instead of calling lower-level utility functions directly."
        )

        raise AssertionError("\n".join(error_messages))


if __name__ == "__main__":
    test_validation_statistics()
    test_validation_missing_tests()
    test_validation_decorator_only_on_computation_functions()
    test_computation_functions_documented_in_features()
    test_validation_tests_call_computation_functions()
