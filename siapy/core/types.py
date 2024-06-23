from PIL.Image import Image
import numpy as np
from siapy.entities import SpectralImage

ImageType = SpectralImage | np.ndarray | Image
