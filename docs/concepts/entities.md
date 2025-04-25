# Entities

Entities serve as the foundational data structures in SiaPy, representing key elements of spectral image analysis and processing workflows. They implement consistent, strongly-typed interfaces that allow seamless interaction between spectral data, spatial coordinates, and geometric information.
Each entity follows a specialized design pattern optimized for its specific role while maintaining compatibility with the broader SiaPy ecosystem.

## Spectral Image

A `SpectralImage` is the primary container for spectral image data. It's a generic class that can wrap different image backends, allowing you to work with various file formats through a unified interface.

### Image Initialization Options

#### 1. Load from ENVI format (using spectral python)

This is commonly used for hyperspectral imagery from airborne or satellite sensors.

```python
--8<-- "docs/concepts/src/spectral_image_01.py"
```

#### 2. Load from GeoTIFF or other geospatial formats (using rasterio)

Perfect for georeferenced data with spatial information.

```python
--8<-- "docs/concepts/src/spectral_image_02.py"
```

#### 3. Create from numpy array

Useful for testing or when you already have image data in memory.

```python
--8<-- "docs/concepts/src/spectral_image_03.py"
```

#### 4. Create your own custom image class

For specialized file formats or custom processing needs, you can extend the ImageBase class.

```python
--8<-- "docs/concepts/src/spectral_image_04.py"
```

## Pixels

The `Pixels` class represents spatial coordinates within spectral image, providing a container for *(x, y)* coordinate pairs. It uses pandas DataFrame internally for storage, enabling high-performance operations. The class provides multiple initialization methods and conversion functions to work with different data representations (i.e. dataframes, list, arrays)

```python
--8<-- "docs/concepts/src/pixels_01.py"
```

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
