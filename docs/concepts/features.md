# Features

??? note "API Documentation"
    `siapy.features`

The features module provides automated feature engineering and selection capabilities specifically designed for spectral data analysis.

## Spectral Indices

??? api "API Documentation"
    [`siapy.features.spectral_indices`][siapy.features.spectral_indices]

Spectral indices are mathematical combinations of spectral bands that highlight specific characteristics of materials or conditions. The module provides functions to discover available indices and compute them from spectral data.

### Getting available indices

The `get_spectral_indices()` function returns all spectral indices that can be computed from the available bands:

```python
--8<-- "docs/concepts/src/features_01.py"
```

### Computing spectral indices

The `compute_spectral_indices()` function calculates spectral indices from DataFrame data:

```python
--8<-- "docs/concepts/src/features_02.py"
```

### Band mapping

When your data uses non-standard column names, use the `bands_map` parameter:

```python
--8<-- "docs/concepts/src/features_03.py:map"
```

## Automatic features generation

??? api "API Documentation"
    [`siapy.features.AutoFeatClassification`][siapy.features.AutoFeatClassification]<br>
    [`siapy.features.AutoFeatRegression`][siapy.features.AutoFeatRegression]<br>
    [`siapy.features.AutoSpectralIndicesClassification`][siapy.features.AutoSpectralIndicesClassification]<br>
    [`siapy.features.AutoSpectralIndicesRegression`][siapy.features.AutoSpectralIndicesRegression]

### Mathematically extracted features

The AutoFeat classes provide deterministic wrappers around the AutoFeat library, which automatically generates and selects engineered features through symbolic regression.

```python
--8<-- "docs/concepts/src/features_04.py"
```

### Features extracted using spectral indices

These classes integrate spectral index computation with automated feature selection, offering end-to-end pipelines for identifying the most relevant spectral indices.

```python
--8<-- "docs/concepts/src/features_05.py"
```

## Integration with siapy enitites

The features module integrates seamlessly with siapy entity system.

```python
--8<-- "docs/concepts/src/features_06.py"
```
