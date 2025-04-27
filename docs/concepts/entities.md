# Entities

Entities serve as the foundational data structures in SiaPy, representing key elements of spectral image analysis and processing workflows. They implement consistent, strongly-typed interfaces that allow seamless interaction between spectral data, spatial coordinates, and geometric information.

## Design principles

SiaPy's architecture follows several key design principles:

**Specialized yet compatible**: Each entity is optimized for its specific role while maintaining compatibility with the broader SiaPy ecosystem

**Independence**: Most entities can function independently (with the exception of abstract base classes)

**Composition over inheritance**:

- The composition is preferred, so that one class can leverage another by injecting it into its architecture
- Inheritance is primarily used to implement common interfaces through base classes

**Extensibility**:

- The `SpectralImage` class supports multiple data sources through *spectral* or *rasterio* libraries, however, custom data loading can be implemented by creating your own driver
- Basic geometric shapes (e.g. points, lines, polygons) are implemented using the *shapely* library, which could also be extended through base abstraction

The class structure is depicted in the following diagram:

![Entities Schematics](images/entities_schematics.png)

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

The `Pixels` class represents spatial coordinates within spectral image, providing a container for *(x, y)* coordinate pairs. It uses pandas DataFrame internally for storage, enabling high-performance operations. The class provides multiple initialization methods and conversion functions to work with different data representations (i.e. DataFrames, list, arrays)

```python
--8<-- "docs/concepts/src/pixels_01.py"
```

## Signals

The `Signals` class stores spectral data for each pixel in a pandas DataFrame, allowing you to use any column names you choose (e.g. "band_1", "nir", "red_edge"). You can initialize it from a DataFrame, lists, dicts or NumPy arrays.

```python
--8<-- "docs/concepts/src/signals_01.py"
```

## Signatures

The `Signatures` class represents spectral data collections by combining spatial coordinates (`Pixels`) with their corresponding spectral values (`Signals`). It provides a unified container that maintains the spatial-spectral relationship, allowing for analysis of spectral information at specific image locations. Internally, the data is stored as pandas DataFrames for efficient operations and indexing.

```python
--8<-- "docs/concepts/src/signatures_01.py"
```

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
