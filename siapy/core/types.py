from typing import Any, Sequence

import numpy as np
import pandas as pd
import spectral as sp
import xarray as xr
from numpy.typing import ArrayLike, NDArray
from PIL.Image import Image

from siapy.entities import SpectralImage, SpectralImageSet

__all__ = [
    "SpectralLibType",
    "XarrayType",
    "ImageType",
    "ImageSizeType",
    "ImageDataType",
    "ImageContainerType",
    "ArrayLike1dType",
    "ArrayLike2dType",
]

SpectralLibType = sp.io.envi.BilFile | sp.io.envi.BipFile | sp.io.envi.BsqFile
XarrayType = xr.DataArray | xr.Dataset
ImageType = SpectralImage[Any] | NDArray[np.floating[Any]] | Image
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
ImageContainerType = SpectralImage[Any] | SpectralImageSet
ArrayLike1dType = NDArray[np.floating[Any]] | pd.Series | Sequence[Any] | ArrayLike
ArrayLike2dType = NDArray[np.floating[Any]] | pd.DataFrame | Sequence[Any] | ArrayLike
