import numpy as np

from siapy.entities import SpectralImage
from siapy.transformations.image import (
    random_crop,
    random_mirror,
    random_rotation,
    rescale,
)

# Create a sample spectral image for demonstration
rng = np.random.default_rng(seed=42)
image_array = rng.random((100, 100, 10))  # height, width, bands
image = SpectralImage.from_numpy(image_array)

# Convert to numpy array for transformations
# image = image.to_numpy() # Not needed, as transformations work directly with SpectralImage

# Resize image to new dimensions
resized_image = rescale(image, (150, 150))
print(f"Original shape: {image.shape}")
print(f"Resized shape: {resized_image.shape}")

# Crop a random section of the image
cropped_image = random_crop(image, (80, 80))
print(f"Cropped shape: {cropped_image.shape}")

# Apply horizontal/vertical mirroring
mirrored_image = random_mirror(image)
print(f"Mirrored shape: {mirrored_image.shape}")

# Rotate image by specified angle (in degrees)
rotated_image = random_rotation(image, angle=30)
print(f"Rotated shape: {rotated_image.shape}")
