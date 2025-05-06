import numpy as np
from rich import print

from siapy.entities import Pixels, SpectralImage

# Create mock image
rng = np.random.default_rng(seed=42)
array = rng.random((100, 100, 10))  # height, width, bands
image = SpectralImage.from_numpy(array)

# Define pixels
iterable = [(1, 2), (3, 4), (2, 4)]
pixels = Pixels.from_iterable(iterable)

# Get signatures
signatures = image.to_signatures(pixels)
print(f"Signatures:\n{signatures}")

# Get numpy array
subarray = image.to_subarray(pixels)
print(f"Subarray:\n{subarray}")
"""
? Note:
    The extracted block has shape (3, 3, 10): a 3 × 3 pixel window across 10 spectral bands.
    Values are populated only at the requested pixel coordinates; all other elements are set to NaN.
"""
