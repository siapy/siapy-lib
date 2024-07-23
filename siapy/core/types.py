import numpy as np
import spectral as sp
from PIL.Image import Image
import pandas as pd
from typing import Sequence, Any

from siapy.entities import SpectralImage, SpectralImageSet

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
