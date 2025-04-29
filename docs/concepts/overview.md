# Library Overview

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
