# Plotting Utilities

??? note "API Documentation"
    `siapy.utils.plots`

The plotting utilities module provides interactive tools for pixel and area selection, as well as visualization functions for spectral images and signals.

## Interactive Pixel Selection

### Point-based Selection

Select individual pixels from an image by clicking on them.

```python
--8<-- "docs/concepts/src/utils_plotting_01.py"
```

### Area-based Selection

Select irregular areas from an image using lasso selection tool.

```python
--8<-- "docs/concepts/src/utils_plotting_02.py"
```

## Image Visualization

### Display Images with Selected Areas

Visualize spectral images with overlaid selected pixels or areas.

```python
--8<-- "docs/concepts/src/utils_plotting_03.py"
```

### Multiple Image Comparison

Display multiple images side by side with their corresponding selected areas.

```python
--8<-- "docs/concepts/src/utils_plotting_04.py"
```

## Signal Visualization

Plot mean spectral signatures with standard deviation bands for different classes.

```python
--8<-- "docs/concepts/src/utils_plotting_05.py"
```
