import numpy as np
from PIL.Image import Image

from siapy.core.types import ImageSizeType, ImageType
from siapy.entities import SpectralImage


def validate_image_to_numpy_3channels(image: ImageType) -> np.ndarray:
    if isinstance(image, SpectralImage):
        image_display = np.array(image.to_display())
    elif isinstance(image, Image):
        image_display = np.array(image)
    elif (
        isinstance(image, np.ndarray) and len(image.shape) == 3 and image.shape[-1] == 3
    ):
        image_display = image.copy()
    else:
        raise ValueError(
            "Argument image must be convertible to numpy array with 3 channels."
        )
    return image_display


def validate_image_to_numpy(image: ImageType) -> np.ndarray:
    if isinstance(image, SpectralImage):
        image_np = image.to_numpy()
    elif isinstance(image, Image):
        image_np = np.array(image)
    elif isinstance(image, np.ndarray):
        image_np = image.copy()
    else:
        raise ValueError("Argument image must be convertible to a numpy array.")
    return image_np


def validate_image_size(output_size: ImageSizeType) -> tuple[int, int]:
    if not isinstance(output_size, (int, tuple)):
        raise TypeError("Argument output_size must be an int or a tuple.")
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    elif len(output_size) != 2 or not all([isinstance(el, int) for el in output_size]):
        raise ValueError(
            "Argument output_size tuple must have 2 elements and contain only integers."
        )
    return output_size
