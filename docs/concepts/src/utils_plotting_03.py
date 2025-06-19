import numpy as np

from siapy.entities import Pixels, SpectralImage
from siapy.utils.plots import display_image_with_areas

# Create a mock spectral image
rng = np.random.default_rng(seed=42)
image_array = rng.random((50, 50, 4))
image = SpectralImage.from_numpy(image_array)

# Create predefined areas manually
area1 = Pixels.from_iterable([(10, 15), (12, 18), (15, 20)])
area2 = Pixels.from_iterable([(i, j) for i in range(20, 25) for j in range(30, 35)])
predefined_areas = [area1, area2]

# Display image with predefined areas
display_image_with_areas(image, predefined_areas, color="white")
