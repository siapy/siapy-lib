import random
from typing import Callable

import numpy as np
from skimage import transform

from siapy.core.types import ImageType
from siapy.utils.general import validate_and_convert_image

OutputSizeType = int | tuple[int, int]


def _check_image_output_size(output_size: OutputSizeType) -> tuple[int, int]:
    if not isinstance(output_size, (int, tuple)):
        raise TypeError("Argument output_size must be an int or a tuple.")
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    elif len(output_size) != 2 or not all([isinstance(el, int) for el in output_size]):
        raise ValueError(
            "Argument output_size tuple must have 2 elements and contain only integers."
        )
    return output_size


def add_gaussian_noise(
    image: ImageType,
    mean: float = 0.0,
    std: float = 1.0,
    clip_to_max: bool = True,
) -> np.ndarray:
    image_np = validate_and_convert_image(image)
    rng = np.random.default_rng(seed=None)
    noise = rng.normal(loc=mean, scale=std, size=image_np.shape)
    image_np = image_np + noise
    if clip_to_max:
        image_np = np.clip(image_np, 0, np.max(image_np))
    return image_np


def random_crop(image: ImageType, output_size: OutputSizeType) -> np.ndarray:
    image_np = validate_and_convert_image(image)
    output_size = _check_image_output_size(output_size)
    h, w = image_np.shape[:2]
    new_h, new_w = output_size
    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)
    return image_np[top : top + new_h, left : left + new_w]


def random_mirror(image: ImageType) -> np.ndarray:
    image_np = validate_and_convert_image(image)
    axis = random.choices([0, 1, (0, 1), None])[0]
    if isinstance(axis, int) or isinstance(axis, tuple):
        image_np = np.flip(image_np, axis=axis)
    return image_np


def random_rotation(image: ImageType, angle: float) -> np.ndarray:
    image_np = validate_and_convert_image(image)
    rotated_image = transform.rotate(image_np, angle)
    return rotated_image


def rescale(image: ImageType, output_size: OutputSizeType) -> np.ndarray:
    image_np = validate_and_convert_image(image)
    output_size = _check_image_output_size(output_size)
    return transform.resize(image_np, output_size)


def area_normalization(image: ImageType) -> np.ndarray:
    image_np = validate_and_convert_image(image)

    def _signal_normalize(signal: np.ndarray) -> np.ndarray:
        area = np.trapz(signal)
        if area == 0:
            return signal
        return signal / area

    def _image_normalization(
        image_np: np.ndarray, func1d: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        return np.apply_along_axis(func1d, axis=2, arr=image_np)

    return _image_normalization(image_np, _signal_normalize)
