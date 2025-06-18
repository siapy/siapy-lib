from pathlib import Path

from siapy.entities import SpectralImage
from siapy.utils.plots import pixels_select_click

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

# Select pixels from the image
pixels = pixels_select_click(image)
# ? Press enter to finish the selection
print("Pixels:", pixels.df)
