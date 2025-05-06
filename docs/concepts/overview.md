# Library overview

## API design principles

SiaPy follows consistent method naming conventions to make the API intuitive and predictable. Understanding these conventions helps you navigate the library more effectively and write code that aligns with the project's style.

**Simple properties**: Noun form for direct attribute access

- Examples: `image.geometric_shapes`, `pixels.df`
- When to use: For quick, computationally inexpensive property access

**Expensive computations**: Prefixed with `get_` for methods that require significant processing

- Examples: `pixels.get_coordinate()`
- When to use: When the operation involves complex calculations or data retrieval

**Alternative constructors**: Prefixed with `from_` for methods that create objects from different data sources

- Examples: `from_numpy()`, `from_dataframe()`, `from_shapefile()`
- When to use: When creating an object from an existing data structure

**Data converters**: Prefixed with `to_` for methods that transform data to another format

- Examples: `to_numpy()`, `to_dataframe()`, `to_geojson()`
- When to use: When exporting data to another representation

**File operations**:

Prefixed with `open_` for reading data from file

- Examples: `open_envi()`, `open_shapefile()`, `open_csv()`

Prefixed with `save_` for writing data to file

- Examples: `save_to_csv()`, `save_to_geotiff()`, `save_as_json()`

**Actions and processing**:

Verbs for operations that modify data or perform calculations

- Examples: `normalize()`, `calculate_ndvi()`, `extract_features()`

Plural forms for batch operations on multiple items

- Examples: `process_images()`, `extract_signatures()`, `calculate_indices()`

**Boolean queries**: Prefixed with `is_`, `has_`, or `can_` for methods returning boolean values

- Examples: `is_valid()`, `has_metadata()`, `can_transform()`
- When to use: For methods that check conditions or properties

**Factory methods**: Prefixed with `create_` for methods that generate new instances

- Examples: `create_mask()`, `create_subset()`, `create_transformer()`
- When to use: When creating new objects based on specific parameters

## Architecture

SiaPy follows a modular architecture organized around key components that work together to provide a comprehensive toolkit for spectral image analysis.

### Core

??? note "API Documentation"
    `siapy.core`

The foundation of the library providing essential functionality:

- **Logging**: Centralized logging functionality
- **Exception handling**: Custom exceptions for consistent error handling
- **Type definitions**: Common types used throughout the library
- **Configuration**: System paths and global configuration settings

### Entities

??? note "API Documentation"
    `siapy.entities`

Fundamental data structures that represent spectral imaging data:

- **Spectral image**: An abstraction for various image formats
- **Spectral image set**: Collection of spectral images with batch operations
- **Pixels**: Representation of pixel coordinates and groups
- **Shapes**: Geometric shapes for images' regions selection and masking
- **Signatures**: Spectral signatures extracted from images

### Datasets

??? note "API Documentation"
    `siapy.datasets`

Tools for managing and working with datasets:

- **Tabular datasets**: Handling tabular data with spectral information

### Features

??? note "API Documentation"
    `siapy.features`

Functionality for working with spectral features:

- **Features**: Abstractions for feature extraction and selection
- **Spectral indices**: Calculation of various spectral indices

### Transformations

??? note "API Documentation"
    `siapy.transformations`

Transformation capabilities:

- **Co-registration**: Aligning images from different sources
- **Image processing**: Functions for image manipulation

### Optimizers

??? note "API Documentation"
    `siapy.optimizers`

Optimization, hyperparameter tuning and evaluation:

- **Optimization**: Machine learning training and optimization of hyperparameters
- **Evaluation metrics and scoring mechanisms**: Tools for assessing model performance

### Utils

??? note "API Documentation"
    `siapy.utils`

Utility and plotting functions:

- **Plotting**: Visualization tools for spectral data
- **Image utilities**: Helper functions for image processing
- **Signature utilities**: Functions for working with spectral signatures
