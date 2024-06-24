import numpy as np
from PIL.Image import Image


from siapy.entities import SpectralImage

ImageType = SpectralImage | np.ndarray | Image
ImageSizeType = int | tuple[int, int]
