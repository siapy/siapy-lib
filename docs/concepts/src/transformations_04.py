import numpy as np

from siapy.entities import Pixels, SpectralImage
from siapy.transformations import corregistrator

# Create two sample spectral images representing different cameras/sensors
rng = np.random.default_rng(seed=42)

# VNIR camera image (visible to near-infrared)
vnir_array = rng.random((100, 120, 50))  # Different dimensions to simulate real cameras
vnir_image = SpectralImage.from_numpy(vnir_array)

# SWIR camera image (short-wave infrared)
swir_array = rng.random((90, 110, 30))
swir_image = SpectralImage.from_numpy(swir_array)

# In practice, you would interactively select corresponding points using:
# from siapy.utils.plots import pixels_select_click
# control_points_vnir = pixels_select_click(vnir_image)
# control_points_swir = pixels_select_click(swir_image)

# For demonstration, create synthetic corresponding points
# These represent the same physical locations viewed by both cameras
control_points_vnir = Pixels.from_iterable([(20, 15), (80, 25), (50, 70), (90, 80), (30, 90)])
control_points_swir = Pixels.from_iterable([(18, 12), (75, 22), (45, 65), (85, 75), (28, 85)])

# Compute transformation matrix that maps VNIR coordinates to SWIR coordinates
transformation_matrix, residual_errors = corregistrator.align(
    control_points_swir,  # Target coordinates (SWIR camera space)
    control_points_vnir,  # Source coordinates (VNIR camera space)
    plot_progress=False,
)

print("Transformation matrix:")
print(transformation_matrix)
print(f"Residual alignment errors: {residual_errors}")
