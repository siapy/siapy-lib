import numpy as np

from siapy.entities import SpectralImage
from siapy.transformations.image import area_normalization

# Create a sample spectral image
rng = np.random.default_rng(seed=42)
image_array = rng.random((100, 100, 10))
image = SpectralImage.from_numpy(image_array)

# Apply area normalization to standardize spectral intensities
# This normalizes each pixel's spectrum by its total area under the curve
normalized_image = area_normalization(image)

# Compare before and after normalization (e.g., pixel at (25, 25))
original_pixel = image.to_numpy()[25, 25, :]
normalized_pixel = normalized_image[25, 25, :]

print(f"Original pixel spectrum area: {np.trapz(original_pixel):.4f}")
print(f"Normalized pixel spectrum area: {np.trapz(normalized_pixel):.4f}")
print(f"Original intensity range: [{original_pixel.min():.3f}, {original_pixel.max():.3f}]")
print(f"Normalized intensity range: [{normalized_pixel.min():.3f}, {normalized_pixel.max():.3f}]")
