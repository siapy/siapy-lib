from typing import Any, Sequence

import numpy as np
import pandas as pd
import spectral as sp
from PIL.Image import Image

from siapy.entities import SpectralImage, SpectralImageSet

__all__ = [
    "SpectralType",
    "ImageType",
    "ImageSizeType",
    "ImageDataType",
    "ImageContainerType",
    "ArrayLike1dType",
    "ArrayLike2dType",
]

SpectralType = sp.io.envi.BilFile | sp.io.envi.BipFile | sp.io.envi.BsqFile
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
ImageContainerType = SpectralImage | SpectralImageSet
ArrayLike1dType = np.ndarray | pd.Series | Sequence[Any]
ArrayLike2dType = np.ndarray | pd.DataFrame | Sequence[Any]
