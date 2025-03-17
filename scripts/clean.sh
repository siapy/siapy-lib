#!/usr/bin/env bash

set -e
set -x

rm -rf $(find . -name __pycache__)
rm -f $(find . -type f -name '*.py[co]')
rm -f $(find . -type f -name '*~')
rm -f $(find . -type f -name '.*~')
rm -rf .pdm-build
rm -rf .mypy_cache
rm -rf .cache
rm -rf .pytest_cache
rm -rf .ruff_cache
rm -rf htmlcov
rm -rf *.egg-info
rm -f .coverage
rm -f .coverage.*
rm -rf build
rm -rf dist
rm -rf site
rm -rf docs/_build
rm -rf coverage.xml
