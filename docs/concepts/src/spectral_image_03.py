import numpy as np

from siapy.entities import SpectralImage

# Create a SpectralImage from a numpy array - mostly for testing
array = np.zeros((100, 100, 10))  # height, width, bands
image = SpectralImage.from_numpy(array)
