from pathlib import Path

import spectral as sp

from siapy.entities import SpectralImage
from siapy.entities.images import SpectralLibImage

# Set the path to the directory containing the data
# !! ADJUST THIS PATH TO YOUR DATA DIRECTORY !!
data_dir = "./docs/examples/data"

# Find all header and image files in the data directory
header_paths = sorted(Path(data_dir).rglob("*.hdr"))
image_paths = sorted(Path(data_dir).rglob("*.img"))

header_path_img0 = header_paths[0]
image_path_img0 = image_paths[0]

# Load the image using spectral library and then wrap over SpectralImage object
sp_file = sp.envi.open(file=header_path_img0, image=image_path_img0)
assert not isinstance(sp_file, sp.io.envi.SpectralLibrary)
image = SpectralImage(SpectralLibImage(sp_file))

# or you can do the same just by running
image = SpectralImage.spy_open(
    header_path=header_path_img0,
    image_path=image_path_img0,
)

# Now you can easily use various property and util functions of the SpectralImage object
# Get the shape of the image
print("Image shape:", image.shape)

# Get the number of bands
print("Number of bands:", image.bands)

# Get the wavelength information
print("Wavelengths:", image.wavelengths)

# Get the file path
print("File path:", image.filepath)

# Get the metadata
print("Metadata:", image.metadata)

# Get the number of rows
print("Number of rows:", image.image.rows)

# Get the number of columns
print("Number of columns:", image.image.cols)

# Get the default bands
print("Default bands:", image.default_bands)

# Get the description
print("Description:", image.image.description)

# Get the camera ID
print("Camera ID:", image.camera_id)

# Get the geometric shapes
print("Geometric shapes:", image.geometric_shapes)
