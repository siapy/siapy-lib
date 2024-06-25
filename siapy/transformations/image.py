import random
from typing import Callable

import numpy as np
from skimage import transform

from siapy.core.types import ImageSizeType, ImageType
from siapy.utils.validators import validate_image_size, validate_image_to_numpy


def add_gaussian_noise(
    image: ImageType,
    mean: float = 0.0,
    std: float = 1.0,
    clip_to_max: bool = True,
) -> np.ndarray:
    image_np = validate_image_to_numpy(image)
    rng = np.random.default_rng()
    noise = rng.normal(loc=mean, scale=std, size=image_np.shape)
    image_np = image_np + noise
    if clip_to_max:
        image_np = np.clip(image_np, 0, np.max(image_np))
    return image_np


def random_crop(image: ImageType, output_size: ImageSizeType) -> np.ndarray:
    image_np = validate_image_to_numpy(image)
    output_size = validate_image_size(output_size)
    h, w = image_np.shape[:2]
    new_h, new_w = output_size
    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)
    return image_np[top : top + new_h, left : left + new_w]


def random_mirror(image: ImageType) -> np.ndarray:
    image_np = validate_image_to_numpy(image)
    axis = random.choices([0, 1, (0, 1), None])[0]
    if isinstance(axis, int) or isinstance(axis, tuple):
        image_np = np.flip(image_np, axis=axis)
    return image_np


def random_rotation(image: ImageType, angle: float) -> np.ndarray:
    image_np = validate_image_to_numpy(image)
    rotated_image = transform.rotate(image_np, angle, preserve_range=True)
    return rotated_image


def rescale(image: ImageType, output_size: ImageSizeType) -> np.ndarray:
    image_np = validate_image_to_numpy(image)
    output_size = validate_image_size(output_size)
    rescaled_image = transform.resize(image_np, output_size, preserve_range=True)
    return rescaled_image


def area_normalization(image: ImageType) -> np.ndarray:
    image_np = validate_image_to_numpy(image)

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
