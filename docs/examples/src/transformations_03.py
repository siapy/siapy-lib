from pathlib import Path

from siapy.entities import SpectralImage
from siapy.transformations.image import (
    add_gaussian_noise,
    area_normalization,
    random_crop,
    random_mirror,
    random_rotation,
    rescale,
)

# Set the path to the directory containing the data
# !! ADJUST THIS PATH TO YOUR DATA DIRECTORY !!
data_dir = "./docs/examples/data"

# Get first image
header_path_img0 = sorted(Path(data_dir).rglob("*.hdr"))[0]
image_path_img0 = sorted(Path(data_dir).rglob("*.img"))[0]

# Load VNIR and SWIR spectral images
image_swir = SpectralImage.spy_open(
    header_path=header_path_img0,
    image_path=image_path_img0,
)

# Convert image to numpy array
image_swir_np = image_swir.to_numpy()

# Apply transformations to image_swir
# Add Gaussian noise
noisy_image = add_gaussian_noise(image_swir_np, mean=0.0, std=1.0, clip_to_max=True)

# Random crop
cropped_image = random_crop(image_swir_np, output_size=(100, 100))

# Random mirror
mirrored_image = random_mirror(image_swir_np)

# Random rotation
rotated_image = random_rotation(image_swir_np, angle=45)

# Rescale
rescaled_image = rescale(image_swir_np, output_size=(200, 200))

# Area normalization
normalized_image = area_normalization(image_swir_np)
