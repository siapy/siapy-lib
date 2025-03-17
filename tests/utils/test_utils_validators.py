import numpy as np
import pytest
from PIL import Image as PILImage
from sklearn.base import BaseEstimator

from siapy.core.exceptions import (
    InvalidInputError,
    InvalidTypeError,
    MethodNotImplementedError,
)
from siapy.utils.validators import (
    check_model_prediction_methods,
    validate_image_size,
    validate_image_to_numpy,
    validate_image_to_numpy_3channels,
)


def test_validate_image_to_numpy_3channels_with_spectral_image(spectral_images):
    image_vnir = spectral_images.vnir
    image_vnir_np = validate_image_to_numpy_3channels(image_vnir)
    assert isinstance(image_vnir_np, np.ndarray)
    assert image_vnir_np.shape[2] == 3
    assert isinstance(validate_image_to_numpy_3channels(image_vnir.to_display()), np.ndarray)


def test_validate_image_to_numpy_3channels_with_image():
    mock_image = PILImage.new("RGB", (100, 100))
    result = validate_image_to_numpy_3channels(mock_image)
    assert isinstance(result, np.ndarray) and result.shape == (100, 100, 3)


def test_validate_image_to_numpy_3channels_with_numpy_array():
    mock_numpy_array = np.random.rand(100, 100, 3)
    assert np.array_equal(validate_image_to_numpy_3channels(mock_numpy_array), mock_numpy_array)


def test_validate_image_to_numpy_3channels_with_invalid_input():
    mock_numpy_array = np.random.rand(100, 100, 4)
    with pytest.raises(InvalidInputError):
        validate_image_to_numpy_3channels(mock_numpy_array)


def test_validate_image_to_numpy_with_spectral_image(spectral_images):
    image_vnir = spectral_images.vnir
    image_vnir_np = validate_image_to_numpy(image_vnir)
    assert isinstance(image_vnir_np, np.ndarray)
    assert image_vnir.shape == image_vnir_np.shape


def test_validate_image_to_numpy_with_pil_image():
    mock_image = PILImage.new("RGB", (100, 100))
    result = validate_image_to_numpy(mock_image)
    assert isinstance(result, np.ndarray) and result.shape == (100, 100, 3)


def test_validate_image_to_numpy_with_numpy_array():
    mock_numpy_array = np.random.rand(100, 100, 5)
    result = validate_image_to_numpy(mock_numpy_array)
    assert np.array_equal(result, mock_numpy_array)


def test_validate_image_to_numpy_with_invalid_input():
    with pytest.raises(InvalidInputError):
        validate_image_to_numpy("invalid_input")


def test_validate_image_size():
    output_size_int = validate_image_size(100)
    assert output_size_int == (100, 100)

    output_size_tuple = validate_image_size((100, 150))
    assert output_size_tuple == (100, 150)

    with pytest.raises(InvalidTypeError):
        validate_image_size("invalid")

    with pytest.raises(InvalidInputError):
        validate_image_size((100,))

    with pytest.raises(InvalidInputError):
        validate_image_size((100, "150"))


class MockModelValid(BaseEstimator):
    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def score(self, X, y):
        pass


class MockModelInvalid(BaseEstimator):
    def fit(self, X, y):
        pass


def test_check_model_prediction_methods_valid():
    model = MockModelValid()
    check_model_prediction_methods(model)


def test_check_model_prediction_methods_invalid():
    model = MockModelInvalid()
    with pytest.raises(
        MethodNotImplementedError,
        # match="The model must have methods: 'fit', 'predict', and 'score'.",
    ):
        check_model_prediction_methods(model)
