import multiprocessing
import types
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest

from siapy.utils.general import (
    dict_zip,
    ensure_dir,
    get_number_cpus,
    initialize_function,
    initialize_object,
    get_increasing_seq_indices,
)


class MockModule(types.ModuleType):
    TestClass = MagicMock(return_value="initialized_object")
    test_function = MagicMock(return_value="function_result")


# Instantiate the mock module
mock_module = MockModule("mock_module")


def test_initialize_object():
    obj = initialize_object(mock_module, "TestClass")
    assert obj == "initialized_object"
    mock_module.TestClass.assert_called_with()

    obj = initialize_object(mock_module, "TestClass", module_args={"arg1": 1}, arg2=2)
    mock_module.TestClass.assert_called_with(arg1=1, arg2=2)

    with pytest.raises(AssertionError):
        initialize_object(mock_module, "TestClass", module_args={"arg1": 1}, arg1=2)


def test_initialize_function():
    func = initialize_function(mock_module, "test_function")
    assert func() == "function_result"
    mock_module.test_function.assert_called_with()

    func = initialize_function(
        mock_module, "test_function", module_args={"arg1": 1}, arg2=2
    )
    assert func() == "function_result"
    mock_module.test_function.assert_called_with(arg1=1, arg2=2)

    with pytest.raises(AssertionError):
        initialize_function(
            mock_module, "test_function", module_args={"arg1": 1}, arg1=2
        )


def test_ensure_dir():
    with TemporaryDirectory() as tmpdirname:
        path = ensure_dir(tmpdirname)
        assert path.is_dir()
    with TemporaryDirectory() as tmpdirname:
        new_dir = Path(tmpdirname) / "new_dir"
        path = ensure_dir(new_dir)
        assert path.is_dir() and path == new_dir


def test_get_number_cpus():
    assert get_number_cpus() == multiprocessing.cpu_count()
    assert get_number_cpus(2) == 2
    assert (
        get_number_cpus(multiprocessing.cpu_count() + 10) == multiprocessing.cpu_count()
    )
    with pytest.raises(ValueError):
        get_number_cpus(0)


def test_dict_zip():
    dict1 = {"a": 1, "b": 2}
    dict2 = {"a": 3, "b": 4}
    zipped = list(dict_zip(dict1, dict2))
    assert zipped == [("a", 1, 3), ("b", 2, 4)]
    dict1 = {"a": 1, "b": 2}
    dict2 = {"a": 3}
    with pytest.raises(ValueError):
        list(dict_zip(dict1, dict2))
    assert list(dict_zip()) == []


def test_get_increasing_seq_indices():
    values_list = [1, 3, 2, 5, 4]
    indices = get_increasing_seq_indices(values_list)
    assert indices == [0, 1, 3]
