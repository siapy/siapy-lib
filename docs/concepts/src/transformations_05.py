import numpy as np

from siapy.entities import Pixels, SpectralImage
from siapy.transformations import corregistrator

# Assume we have the transformation matrix from the previous example
transformation_matrix = np.array(
    [
        [9.53012482e-01, 1.36821892e-03, -1.33929429e00],
        [6.24099856e-04, 9.68410946e-01, -2.46471435e00],
        [2.20958871e-17, -2.81409517e-18, 1.00000000e00],
    ]
)

# Create a VNIR image for demonstration
rng = np.random.default_rng(seed=42)
vnir_array = rng.random((100, 120, 50))
vnir_image = SpectralImage.from_numpy(vnir_array)

# Define pixels of interest in the VNIR image
# These could be selected regions or any features of interest
vnir_pixels = Pixels.from_iterable([(25, 30), (45, 50), (65, 70), (80, 85)])

# Transform these pixels to SWIR camera coordinate system
swir_pixels = corregistrator.transform(vnir_pixels, transformation_matrix)

print("Original VNIR coordinates:")
print(vnir_pixels.df)
print("\nTransformed SWIR coordinates:")
print(swir_pixels.df)


# Applications of transformed coordinates:

# 1. Extract spectral signatures from corresponding locations in both images
#    vnir_signatures = vnir_image.get_signatures_from_pixels(vnir_pixels)
#    swir_signatures = swir_image.get_signatures_from_pixels(swir_pixels)

# 2. Perform cross-spectral analysis of identical physical regions
#    This enables comparison of vegetation indices, material classification,
#    or spectral feature analysis across different wavelength ranges

# 3. Create composite analyses combining VNIR and SWIR data
#    Combined data provides enhanced discrimination capabilities for:
#    - Mineral identification (clay, carbonate, sulfate detection)
#    - Vegetation health assessment (water content, chlorophyll)
#    - Material classification with improved accuracy
