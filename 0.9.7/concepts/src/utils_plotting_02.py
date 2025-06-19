import numpy as np

from siapy.entities import SpectralImage
from siapy.utils.plots import pixels_select_lasso

# Create a mock spectral image
rng = np.random.default_rng(seed=42)
image_array = rng.random((50, 50, 4))
image = SpectralImage.from_numpy(image_array)

# Interactive area selection with custom selector properties
selector_props = {"color": "blue", "linewidth": 3, "linestyle": "--"}

# Draw lasso shapes around areas of interest, then press Enter to finish
selected_areas = pixels_select_lasso(image, selector_props=selector_props)

print(f"Selected {len(selected_areas)} areas:")
for i, area in enumerate(selected_areas):
    print(f"Area {i}: {len(area)} pixels")
    print(f"  Sample coordinates: {area.df.head()}")

# Each area is a separate Pixels object containing all coordinates within the lasso
total_pixels = sum(len(area) for area in selected_areas)
print(f"Total pixels selected across all areas: {total_pixels}")
