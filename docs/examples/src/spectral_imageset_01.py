from pathlib import Path

from siapy.entities import SpectralImageSet

# Set the path to the directory containing the data
# !! ADJUST THIS PATH TO YOUR DATA DIRECTORY !!
data_dir = "./docs/examples/data"

# Find all header and image files in the data directory
header_paths = sorted(Path(data_dir).rglob("*.hdr"))
image_paths = sorted(Path(data_dir).rglob("*.img"))

# Create a SpectralImageSet from the found paths
imageset = SpectralImageSet.spy_open(
    header_paths=header_paths,
    image_paths=image_paths,
)

# Now you can easily use various properties and utility functions of the SpectralImageSet object.
# First, let's sort the images:
print("Unsorted: ", imageset.images)
imageset.sort()
print("Sorted: ", imageset.images)

# Get the number of images in the set
print("Number of images in the set:", len(imageset))

# Get the cameras ID
print("Cameras ID:", imageset.cameras_id)

# Iterate over images and print their shapes
for idx, image in enumerate(imageset):
    print(f"Image {idx} shape:", image.shape)

# Get images by camera ID
camera_id = imageset.cameras_id[0]
images_by_camera = imageset.images_by_camera_id(camera_id)
print(f"Number of images by camera {camera_id}:", len(images_by_camera))
