import numpy as np

from siapy.entities import SpectralImage
from siapy.utils.plots import pixels_select_click

# Create a mock spectral image with 4 bands
rng = np.random.default_rng(seed=42)
image_array = rng.random((50, 50, 4))  # height, width, bands
image = SpectralImage.from_numpy(image_array)

# Interactive pixel selection
# Click on pixels in the displayed image, then press Enter to finish
selected_pixels = pixels_select_click(image)

print(f"Selected {len(selected_pixels)} pixels:")
print(selected_pixels.df)
