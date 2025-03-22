from pathlib import Path

from siapy.entities import SpectralImage
from siapy.transformations import corregistrator
from siapy.utils.plots import pixels_select_click

# Set the path to the directory containing the data
# !! ADJUST THIS PATH TO YOUR DATA DIRECTORY !!
data_dir = "./docs/examples/data"

# Get first image
header_path_img0 = sorted(Path(data_dir).rglob("coregister*corr2_rad_f32.hdr"))[0]
image_path_img0 = sorted(Path(data_dir).rglob("coregister*corr2_rad_f32.img"))[0]
header_path_img1 = sorted(Path(data_dir).rglob("coregister*corr_rad_f32.hdr"))[0]
image_path_img1 = sorted(Path(data_dir).rglob("coregister*corr_rad_f32.img"))[0]

# Load VNIR and SWIR spectral images
image_swir = SpectralImage.spy_open(
    header_path=header_path_img0,
    image_path=image_path_img0,
)
image_vnir = SpectralImage.spy_open(
    header_path=header_path_img1,
    image_path=image_path_img1,
)

# Select the same pixels in both images.
# The more points you select, the better the transformation between image spaces will be.
# Click enter to finish the selection.
pixels_vnir = pixels_select_click(image_vnir)
pixels_swir = pixels_select_click(image_swir)

# Perform the transformation and transform the selected pixels from the VNIR image to the space of the SWIR image.
matx, _ = corregistrator.align(pixels_swir, pixels_vnir, plot_progress=False)
print("Transformation matrix:", matx)
