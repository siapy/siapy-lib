# Transformations

??? note "API Documentation"
    `siapy.transformations`

The transformations module provides essential image processing and co-registration capabilities. It includes functions for spatial image manipulation, data augmentation, and geometric transformations between different camera coordinate systems.

## Image transformations

??? api "API Documentation"
    `siapy.transformations.image`

### Basic transformations

```python
--8<-- "docs/concepts/src/transformations_01.py"
```

### Data augmentation

Data augmentation transformations are useful for expanding training datasets and testing algorithm robustness.

```python
--8<-- "docs/concepts/src/transformations_02.py"
```

### Normalization

The `area_normalization` function normalizes spectral signals by their area under the curve, which is particularly useful for comparing spectral shapes regardless of overall intensity.

```python
--8<-- "docs/concepts/src/transformations_03.py"
```

## Co-registration

??? api "API Documentation"
    `siapy.transformations.corregistrator`

Co-registration enables alignment and coordinate transformation between different spectral images, particularly useful when working with multiple cameras or sensors viewing the same scene.

### Alignment workflow

The typical co-registration workflow involves selecting corresponding points in both images and computing a transformation matrix:

```python
--8<-- "docs/concepts/src/transformations_04.py"
```

### Applying transformations

Once you have a transformation matrix, you can transform pixel coordinates between image spaces:

```python
--8<-- "docs/concepts/src/transformations_05.py"
```
