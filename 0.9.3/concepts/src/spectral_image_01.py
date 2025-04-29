from siapy.entities import SpectralImage
from siapy.entities.images import SpectralLibImage

# Load from ENVI format (uses spectral python library)
image_sp = SpectralLibImage.open(
    header_path="path/to/header.hdr",
    image_path="path/to/image.img",
)
image = SpectralImage(image_sp)

# Or you can use the class method to load the image directly
image = SpectralImage.spy_open(
    header_path="path/to/header.hdr",
    image_path="path/to/image.img",
)
