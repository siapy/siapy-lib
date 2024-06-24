import numpy as np
from PIL.Image import Image

from siapy.entities import SpectralImage

ImageType = SpectralImage | np.ndarray | Image
ImageSizeType = int | tuple[int, int]
ImageDataType = (
    np.uint8
    | np.int16
    | np.int32
    | np.float32
    | np.float64
    | np.complex64
    | np.complex128
    | np.uint16
    | np.uint32
    | np.int64
    | np.uint64
)
