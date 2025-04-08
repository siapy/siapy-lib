# SiaPy Style Guide

## Type Hints & Python Version

- Use Python 3.10+ type hints (prefer `x | y` over `Union[x, y]`)
- Include type hints for all functions and class members

## Code Style

- 4-space indentation (PEP 8)
- No docstrings needed
- Define `__all__` directly under imports

## Class Design

- Use dataclasses for data-oriented classes
- Use pydantic for data validation
- Prefer property decorators for read-only access

## Method Naming

- Properties: simple noun (e.g. `name`, `value`)
- Expensive getters: prefix with `get_` (e.g. `get_statistics()`)
- Constructors: prefix with `from_` (e.g. `from_point()`)
- Data converters: prefix with `to_` (e.g. `to_numpy()`)
- File loaders: prefix with `open_` (e.g. `open_shapefile()`)
- File savers: prefix with `save_` (e.g. `save_to_csv()`)
- Actions/operations: use verbs (e.g. `calculate()`, `process()`)
- Batch operations: use plural nouns (e.g. `process_items()`)
- Boolean queries: prefix with `is_`, `has_` or `can_` (e.g. `is_valid()`, `has_data()`)
- Factory methods: prefix with `create_` (e.g. `create_instance()`)

## Class Organization

1. Dunder methods
2. Class methods
3. Properties
4. Instance methods

## Error Handling

- Use custom exceptions from `siapy.core.exceptions`:
  - InvalidFilepathError
  - InvalidInputError
  - InvalidTypeError
  - ProcessingError
  - ConfigurationError
  - MethodNotImplementedError
  - DirectInitializationError

## Other

- Use `from siapy.core import logger` for logging
- Use Protocol/ABC for interfaces where appropriate
