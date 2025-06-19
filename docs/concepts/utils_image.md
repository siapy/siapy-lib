# Image Utilities

??? note "API Documentation"
    `siapy.utils.images`

The image utilities module provides functions for saving, loading, and processing spectral images with support for both Spectral Python (SPy) and Rasterio backends.

## Image I/O Operations

### Spectral Python backend

The SPy backend saves images in ENVI format.

```python
--8<-- "docs/concepts/src/utils_image_01.py"
```

### Rasterio backend

The Rasterio backend provides geospatial capabilities and supports various formats.

```python
--8<-- "docs/concepts/src/utils_image_02.py"
```

## Radiance to Reflectance Conversion

Converting radiance measurements to reflectance using reference panels is essential for quantitative spectral analysis.

```python
--8<-- "docs/concepts/src/utils_image_03.py"
```

## Additional Utility Functions

```python
--8<-- "docs/concepts/src/utils_image_04.py"
```
