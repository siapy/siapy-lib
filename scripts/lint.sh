#!/usr/bin/env bash

set -e
set -x

pdm run mypy siapy
pdm run ruff check siapy
pdm run ruff format siapy --check
