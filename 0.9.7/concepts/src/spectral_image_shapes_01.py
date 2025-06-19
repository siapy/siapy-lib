# --8<-- [start:init]
import numpy as np
from rich import print

from siapy.entities import Shape, SpectralImage
from siapy.entities.shapes import GeometricShapes

# Create a mock spectral image
rng = np.random.default_rng(seed=42)
array = rng.random((100, 100, 10))  # height, width, bands
image = SpectralImage.from_numpy(array)

# SpectralImage automatically initializes GeometricShapes
assert isinstance(image.geometric_shapes, GeometricShapes)
# You can access the underlying list via the shapes property
assert isinstance(image.geometric_shapes.shapes, list)

# GeometricShapes implements common list operations directly, i.e. number of shapes:
length_via_geometric_shapes = len(image.geometric_shapes)
length_via_raw_list = len(image.geometric_shapes.shapes)
# --8<-- [end:init]

# --8<-- [start:operations]
# Create two Shape objects with distinct spatial coordinates and semantic labels
coords1 = [(1, 2), (3, 4), (2, 4)]
shape1 = Shape.from_multipoint(coords1, label="coords1")
coords2 = [(19, 20), (21, 22), (20, 22)]
shape2 = Shape.from_multipoint(coords2, label="coords2")

# Add multiple shapes to the SpectralImage's geometric_shapes container at once
image.geometric_shapes.extend([shape1, shape2])

# Display the current collection of shapes stored in the image
# Each shape will be shown with its type and label
print(f"Shapes in GeometricShapes: {image.geometric_shapes}")

# --8<-- [end:operations]
