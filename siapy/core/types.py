from typing import Any, Sequence

import numpy as np
import pandas as pd
import spectral as sp
import xarray as xr
from numpy.typing import ArrayLike
from PIL.Image import Image

from siapy.entities import SpectralImage, SpectralImageSet

__all__ = [
    "SpectralLibType",
    "ImageType",
    "ImageSizeType",
    "ImageDataType",
    "ImageContainerType",
    "ArrayLike1dType",
    "ArrayLike2dType",
]

SpectralLibType = sp.io.envi.BilFile | sp.io.envi.BipFile | sp.io.envi.BsqFile
XarrayType = xr.DataArray | xr.Dataset
ImageType = SpectralImage | np.ndarray | Image
ImageSizeType = int | tuple[int, ...]
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
ArrayLike1dType = np.ndarray | pd.Series | Sequence[Any] | ArrayLike
ArrayLike2dType = np.ndarray | pd.DataFrame | Sequence[Any] | ArrayLike
