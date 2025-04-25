# Entities

Entities form the core data structures in SiaPy that represent fundamental elements of spectral image processing. They provide consistent interfaces for working with various kinds of spectral data and spatial information.

## Overview

The SiaPy Entities module defines a set of interconnected classes that represent the foundational building blocks for spectral image analysis:

- **SpectralImage**: A generic container for different types of hyperspectral/multispectral images
- **SpectralImageSet**: A collection of spectral images
- **Pixels**: Spatial coordinates within an image
- **Signatures**: Spectral signals associated with specific pixel locations
- **Shape**: Geometric shapes associated with image locations (points, lines, polygons)

These entity classes are designed to work together, forming a cohesive system for analyzing spectral imagery.

## SpectralImage

A `SpectralImage` is the primary container for spectral image data. It's a generic class that can wrap different image backends:

```python
from siapy.entities import SpectralImage

# Load from ENVI format
image = SpectralImage.spy_open(
    header_path="path/to/header.hdr",
    image_path="path/to/image.img"
)

# Load from GeoTIFF or other raster formats
image = SpectralImage.rasterio_open(filepath="path/to/image.tif")

# Create from NumPy array
import numpy as np
array = np.zeros((100, 100, 10))  # height, width, bands
image = SpectralImage.from_numpy(array)
```

### Key Properties

- **shape**: Dimensions as (height, width, bands)
- **width**, **height**: Image dimensions
- **bands**: Number of spectral bands
- **wavelengths**: List of wavelength values for each band
- **default_bands**: Default bands for RGB visualization
- **metadata**: Dictionary of metadata from the image file
- **filepath**: Path to the source file
- **camera_id**: Camera identifier (if available)
- **geometric_shapes**: Associated geometric shapes collection

### Key Methods

- **to_numpy()**: Convert to NumPy array
- **to_display()**: Convert to PIL Image for visualization
- **to_xarray()**: Convert to xarray.DataArray
- **to_signatures()**: Extract signatures at specified pixels
- **to_subarray()**: Extract a subarray for a region of interest
- **average_intensity()**: Calculate mean intensity across specified axes

## Pixels

The `Pixels` class represents spatial coordinates within an image, typically stored as x,y pairs.

```python
from siapy.entities import Pixels

# Create from list of coordinates
pixels = Pixels.from_iterable([(10, 20), (30, 40), (50, 60)])

# Load from parquet file
pixels = Pixels.load_from_parquet("pixels.parquet")
```

### Key Properties

- **df**: Underlying pandas DataFrame with x,y coordinates
- **coords**: Coordinate system definition

### Key Methods

- **x()**, **y()**: Access x and y coordinates as pandas Series
- **to_numpy()**: Convert to NumPy array
- **to_list()**: Convert to list of coordinates
- **as_type()**: Convert coordinates to a specific data type
- **get_coordinate()**: Get a specific coordinate pair
- **df_homogenious()**: Get homogeneous coordinates (x,y,1)

## Signatures

The `Signatures` class combines `Pixels` with their corresponding spectral signals.

```python
from siapy.entities import Signatures, Pixels, Signals

# Create from pixels and signals
pixels = Pixels.from_iterable([(10, 20), (30, 40)])
signals_df = pd.DataFrame([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])  # 2 pixels, 3 bands
signals = Signals(signals_df)
signatures = Signatures(pixels, signals)

# Extract signatures from an image at specific pixels
pixels = Pixels.from_iterable([(10, 20), (30, 40)])
signatures = spectral_image.to_signatures(pixels)
```

### Key Properties

- **pixels**: The `Pixels` object with coordinate information
- **signals**: The `Signals` object with spectral values

### Key Methods

- **to_dataframe()**: Convert to a pandas DataFrame
- **to_dataframe_multiindex()**: Convert to a DataFrame with MultiIndex columns
- **to_numpy()**: Convert to tuple of NumPy arrays (pixels, signals)
- **to_dict()**: Convert to dictionary representation
- **reset_index()**: Reset DataFrame indices
- **copy()**: Create a deep copy

## Signals

The `Signals` class represents spectral values associated with pixels.

```python
from siapy.entities.signatures import Signals

# Create from DataFrame
signals_df = pd.DataFrame([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])  # 2 pixels, 3 bands
signals = Signals(signals_df)

# Create from iterable
signals = Signals.from_iterable([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
```

### Key Properties

- **df**: Underlying pandas DataFrame with spectral values

### Key Methods

- **to_numpy()**: Convert to NumPy array
- **average_signal()**: Calculate mean signal across specified axis
- **save_to_parquet()**: Save to parquet file

## Shape

The `Shape` class represents geometric shapes that can be associated with images, such as points, lines, and polygons.

```python
from siapy.entities import Shape
from siapy.entities import Pixels

# Create a point
point = Shape.from_point(10, 20)

# Create a polygon from pixels
pixels = Pixels.from_iterable([(0, 0), (10, 0), (10, 10), (0, 10)])
polygon = Shape.from_polygon(pixels)

# Load from shapefile
shape = Shape.open_shapefile("path/to/shapefile.shp")
```

### Shape Types

- **Point**: Single coordinate point (x,y)
- **LineString**: Series of connected points forming a line
- **Polygon**: Closed shape with interior area
- **MultiPoint**: Collection of independent points
- **MultiLineString**: Collection of independent lines
- **MultiPolygon**: Collection of independent polygons

### Key Properties

- **geometry**: The underlying shapely geometry
- **label**: Optional label for the shape
- **shape_type**: Type of geometry (point, line, polygon, etc.)
- **bounds**: Bounding box of the shape
- **centroid**: Centroid point of the shape

### Key Methods

- **buffer()**: Create a buffered version of the shape
- **intersection()**: Find intersection with another shape
- **union()**: Combine with another shape
- **to_file()**: Save to a shapefile

## GeometricShapes

The `GeometricShapes` class manages a collection of shapes associated with a spectral image.

```python
# Access shapes associated with an image
image = SpectralImage.spy_open(header_path="...", image_path="...")
shapes = image.geometric_shapes

# Add a shape
polygon = Shape.from_rectangle(10, 20, 30, 40)
shapes.append(polygon)

# Find a shape by name
shape = shapes.get_by_name("vegetation")
```

### Key Methods

- **append()**, **extend()**: Add shapes to the collection
- **remove()**, **pop()**, **clear()**: Remove shapes
- **index()**, **count()**: Find and count shapes
- **get_by_name()**: Find a shape by its label

## SpectralImageSet

The `SpectralImageSet` class manages a collection of spectral images.

```python
from siapy.entities import SpectralImageSet
from pathlib import Path

# Load multiple images
header_paths = list(Path("data_dir").glob("*.hdr"))
image_paths = list(Path("data_dir").glob("*.img"))
image_set = SpectralImageSet.spy_open(
    header_paths=header_paths,
    image_paths=image_paths
)

# Access images
first_image = image_set[0]

# Sort images
image_set.sort()
```

### Key Properties

- **images**: List of SpectralImage objects
- **cameras_id**: List of unique camera IDs

### Key Methods

- **images_by_camera_id()**: Get images from a specific camera
- **sort()**: Sort the images

## Relationship Between Entities

The entities in SiaPy form a cohesive system:

1. A `SpectralImage` contains pixel data across multiple spectral bands
2. `Pixels` represent spatial coordinates within that image
3. `Signals` contain spectral values at those coordinates
4. `Signatures` combine pixels and signals to represent spectral signatures at specific locations
5. `Shape` objects define geometric regions in the image
6. `GeometricShapes` organize multiple shapes associated with an image
7. `SpectralImageSet` manages multiple related spectral images

This modular design allows for flexible workflows in spectral image analysis - from loading image data, to selecting regions of interest, to extracting and analyzing spectral signatures.
