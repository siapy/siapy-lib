import numpy as np

from siapy.utils.images import spy_create_image, spy_save_image

# Create sample data
rng = np.random.default_rng(42)
image = rng.random((100, 150, 50))  # height, width, bands

# Save image in ENVI format
spy_save_image(
    image=image,
    save_path="output/my_image.hdr",
    metadata={"description": "Sample hyperspectral image"},
    overwrite=True,
    dtype=np.float32,
)

# Create image and get SpectralImage object back
spectral_img = spy_create_image(
    image=image,
    save_path="output/created_image.hdr",
    metadata={
        "lines": image.shape[0],
        "samples": image.shape[1],
        "bands": image.shape[2],
        "wavelength": [400 + i * 5 for i in range(image.shape[2])],
    },
)
