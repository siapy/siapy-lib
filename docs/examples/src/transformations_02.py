from pathlib import Path

import numpy as np

from siapy.entities import SpectralImage
from siapy.transformations import corregistrator
from siapy.utils.plots import display_multiple_images_with_areas, pixels_select_lasso

# Set the path to the directory containing the data
# !! ADJUST THIS PATH TO YOUR DATA DIRECTORY !!
data_dir = "./docs/examples/data"

# Get first image
header_path_img0 = sorted(Path(data_dir).rglob("*.hdr"))[0]
image_path_img0 = sorted(Path(data_dir).rglob("*.img"))[0]
header_path_img1 = sorted(Path(data_dir).rglob("*.hdr"))[1]
image_path_img1 = sorted(Path(data_dir).rglob("*.img"))[1]

# Load VNIR and SWIR spectral images
image_swir = SpectralImage.spy_open(
    header_path=header_path_img0,
    image_path=image_path_img0,
)
image_vnir = SpectralImage.spy_open(
    header_path=header_path_img1,
    image_path=image_path_img1,
)

# Transformation matrix was calculated in previous example
matx = np.array(
    [
        [5.10939099e-01, -3.05286868e-03, -1.48283389e00],
        [-2.15777211e-03, 5.17836773e-01, -2.50694723e01],
        [3.02412467e-18, 7.36518494e-18, 1.00000000e00],
    ]
)

# Select area of the image
# Click enter to finish the selection.
selected_areas_vnir = pixels_select_lasso(image_vnir)

# Transform the selected areas from the VNIR image to the space of the SWIR image.
selected_areas_swir = [corregistrator.transform(pixels_vnir, matx) for pixels_vnir in selected_areas_vnir]

# Display the selected areas in both images
display_multiple_images_with_areas(
    [
        (image_vnir, selected_areas_vnir),
        (image_swir, selected_areas_swir),
    ],
    plot_interactive_buttons=False,
)
