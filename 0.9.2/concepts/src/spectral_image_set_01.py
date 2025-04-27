from pathlib import Path

from siapy.entities import SpectralImageSet

# Load multiple images
header_paths = list(Path("data_dir").glob("*.hdr"))
image_paths = list(Path("data_dir").glob("*.img"))
image_set = SpectralImageSet.spy_open(header_paths=header_paths, image_paths=image_paths)

# Iterate over the images
for image in image_set:
    print(image)
