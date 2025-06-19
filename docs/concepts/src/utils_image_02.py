import numpy as np

from siapy.utils.images import rasterio_create_image, rasterio_save_image

# Create sample data
rng = np.random.default_rng(42)
image = rng.random((100, 150, 5))

# Save as GeoTIFF with Rasterio
rasterio_save_image(
    image=image,
    save_path="output/rasterio_image.tif",
    metadata={"description": "Hyperspectral data", "wavelength": [400, 450, 500, 550, 600]},
)

# Create image and get SpectralImage object back
spectral_img = rasterio_create_image(
    image=image,
    save_path="output/created_rasterio.tif",
    metadata={"wavelength": [400, 450, 500, 550, 600]},
)
