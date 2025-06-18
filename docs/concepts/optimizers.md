# Optimizers

??? note "API Documentation"
    `siapy.optimizers`

The optimizers module provides hyperparameter optimization capabilities for machine learning models used in spectral data analysis. It integrates with Optuna for efficient hyperparameter search and includes other evaluation tools.

## Tabular Optimizer

??? api "API Documentation"
    [`siapy.optimizers.optimizers.TabularOptimizer`][siapy.optimizers.optimizers.TabularOptimizer]<br>
    [`siapy.optimizers.configs.TabularOptimizerConfig`][siapy.optimizers.configs.TabularOptimizerConfig]

The `TabularOptimizer` class provides automated hyperparameter optimization for sklearn-compatible models using tabular spectral data.

### Basic optimization setup

```python
--8<-- "docs/concepts/src/optimizers_01.py"
```

## Trial Parameters

??? api "API Documentation"
    [`siapy.optimizers.parameters.TrialParameters`][siapy.optimizers.parameters.TrialParameters]<br>
    [`siapy.optimizers.parameters.IntParameter`][siapy.optimizers.parameters.IntParameter]<br>
    [`siapy.optimizers.parameters.FloatParameter`][siapy.optimizers.parameters.FloatParameter]<br>
    [`siapy.optimizers.parameters.CategoricalParameter`][siapy.optimizers.parameters.CategoricalParameter]

Trial parameters define the hyperparameter search space for optimization. You can specify integer, float, and categorical parameters:

```python
--8<-- "docs/concepts/src/optimizers_02.py"
```

<!--
## Scorers

??? api "API Documentation"
    [`siapy.optimizers.Scorer`][siapy.optimizers.Scorer]

Scorers define how model performance is evaluated during optimization. The module provides two main scoring strategies:

### Cross-validation scorer

Use cross-validation for robust model evaluation:

```python
--8<-- "docs/concepts/src/optimizers_05.py"
```

### Hold-out scorer

Use hold-out validation for faster evaluation:

```python
--8<-- "docs/concepts/src/optimizers_06.py"
```

## Evaluation Functions

??? api "API Documentation"
    [`siapy.optimizers.cross_validation`][siapy.optimizers.cross_validation]
    [`siapy.optimizers.hold_out_validation`][siapy.optimizers.hold_out_validation]

The evaluation functions can be used independently for model assessment:

### Cross-validation

```python
--8<-- "docs/concepts/src/optimizers_07.py"
```

### Hold-out validation

```python
--8<-- "docs/concepts/src/optimizers_08.py"
```

## Metrics

??? api "API Documentation"
    [`siapy.optimizers.calculate_classification_metrics`][siapy.optimizers.calculate_classification_metrics]
    [`siapy.optimizers.calculate_regression_metrics`][siapy.optimizers.calculate_regression_metrics]
    [`siapy.optimizers.ClassificationMetrics`][siapy.optimizers.ClassificationMetrics]
    [`siapy.optimizers.RegressionMetrics`][siapy.optimizers.RegressionMetrics]

Calculate comprehensive metrics for model evaluation:

### Classification metrics

```python
--8<-- "docs/concepts/src/optimizers_09.py"
```

### Regression metrics

```python
--8<-- "docs/concepts/src/optimizers_10.py"
```

## Integration with siapy entities

The optimizers module integrates seamlessly with the siapy entity system:

```python
--8<-- "docs/concepts/src/optimizers_11.py"
``` -->
