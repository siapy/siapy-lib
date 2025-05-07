import numpy as np
from rich import print

from siapy.datasets.tabular import TabularDataset
from siapy.entities import Shape, SpectralImage, SpectralImageSet

# Create two mock spectral images with dimensions 100x100 pixels and 10 bands
rng = np.random.default_rng(seed=42)
image1 = SpectralImage.from_numpy(rng.random((100, 100, 10)))
image2 = SpectralImage.from_numpy(rng.random((100, 100, 10)))

# Define a region of interest as a rectangular shape (coordinates in pixel space)
# This will select pixels from position (50,50) to (80,80) in both images
rectangle = Shape.from_rectangle(x_min=50, y_min=50, x_max=80, y_max=80)
image1.geometric_shapes.append(rectangle)
image2.geometric_shapes.append(rectangle)

# Combine the images into a SpectralImageSet for batch processing
image_set = SpectralImageSet([image1, image2])

# Initialize the TabularDataset with our image set and process the data
# This extracts pixel data from the regions defined by our shapes
dataset = TabularDataset(image_set)
dataset.process_image_data()

# The dataset is now iterable - we can access each entity (processed region)
for entity in dataset:
    print(f"Processing entity: {entity}")

# Generate tabular data from our processed dataset
# Setting mean_signatures=False preserves individual pixel values instead of averaging them
dataset_data = dataset.generate_dataset_data(mean_signatures=False)

# Convert the tabular data to a pandas DataFrame with multi-level indexing
# This makes it easy to analyze and manipulate the data
df = dataset_data.to_dataframe_multiindex()

print(f"dataset_data: \n{dataset_data}")
print(f"df: \n{df}")
