from pathlib import Path

from siapy.entities import SpectralImage
from siapy.utils.plots import pixels_select_lasso

# Set the path to the directory containing the data
# !! ADJUST THIS PATH TO YOUR DATA DIRECTORY !!
data_dir = "./docs/examples/data"

# Get arbitrary image
header_path_img0 = sorted(Path(data_dir).rglob("*.hdr"))[1]
image_path_img0 = sorted(Path(data_dir).rglob("*.img"))[1]

# Load spectral image
image = SpectralImage.spy_open(
    header_path=header_path_img0,
    image_path=image_path_img0,
)

# Select areas from the image
areas = pixels_select_lasso(image)
# ? Press enter to finish the selection

# Print the selected areas
for i, area in enumerate(areas):
    print(f"Area {i}:", area)
