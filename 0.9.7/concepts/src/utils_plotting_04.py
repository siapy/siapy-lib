import numpy as np

from siapy.entities import Pixels, SpectralImage
from siapy.utils.plots import InteractiveButtonsEnum, display_multiple_images_with_areas

# Create two mock spectral images (e.g., simulating VNIR and SWIR)
rng = np.random.default_rng(seed=42)

# VNIR-like image (4 bands)
vnir_array = rng.random((50, 50, 4))
vnir_image = SpectralImage.from_numpy(vnir_array)

# SWIR-like image (6 bands)
swir_array = rng.random((50, 50, 6))
swir_image = SpectralImage.from_numpy(swir_array)

# Define corresponding areas for both images
vnir_areas = [
    Pixels.from_iterable([(10, 15), (12, 18), (15, 20)]),
    Pixels.from_iterable([(25, 30), (28, 32), (30, 35)]),
]

swir_areas = [
    Pixels.from_iterable([(9, 14), (11, 17), (14, 19)]),  # Slightly offset coordinates
    Pixels.from_iterable([(24, 29), (27, 31), (29, 34)]),
]

# Display multiple images with interactive buttons
result = display_multiple_images_with_areas(
    images_with_areas=[
        (vnir_image, vnir_areas),
        (swir_image, swir_areas),
    ],
    color="white",
    plot_interactive_buttons=True,
)

# Handle user interaction result
if result == InteractiveButtonsEnum.SAVE:
    print("User chose to save the selection")
elif result == InteractiveButtonsEnum.REPEAT:
    print("User chose to repeat the process")
elif result == InteractiveButtonsEnum.SKIP:
    print("User chose to skip this step")
