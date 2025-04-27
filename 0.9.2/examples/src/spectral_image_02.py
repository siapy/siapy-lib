from pathlib import Path

import matplotlib.pyplot as plt

from siapy.entities import Pixels, SpectralImage

# Set the path to the directory containing the data
# !! ADJUST THIS PATH TO YOUR DATA DIRECTORY !!
data_dir = "./docs/examples/data"

# Get first image
header_path_img0 = sorted(Path(data_dir).rglob("*.hdr"))[1]
image_path_img0 = sorted(Path(data_dir).rglob("*.img"))[1]

# Load spectral image
image = SpectralImage.spy_open(
    header_path=header_path_img0,
    image_path=image_path_img0,
)

# Convert to numpy
image_np = image.to_numpy(nan_value=0.0)
print("Image shape:", image_np.shape)

# Calculate mean
mean_val = image.average_intensity(axis=(0, 1))
print("Mean value per band:", mean_val)

# Create a Pixels object from an iterable with pixels coordinates
# The iterable should be a list of tuples representing (x, y) coordinates
# iterable == [(x1, y1), (x2, y2), ...] -> list of pixels
iterable = [(1, 2), (3, 4), (5, 6)]
pixels = Pixels.from_iterable(iterable)

# Convert the pixel coordinates to spectral signatures
signatures = image.to_signatures(pixels)
print("Signatures:", signatures)

# Extract a subarray from the image using the pixel coordinates
subarray = image.to_subarray(pixels)
print("Subarray shape:", subarray.shape)

# Convert to displayable image
display_image = image.to_display(equalize=True)

# Display the image using matplotlib
plt.figure()
plt.imshow(display_image)
plt.axis("off")  # Hide axes for better visualization
plt.show()
