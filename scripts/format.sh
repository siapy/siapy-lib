#!/usr/bin/env bash

set -e
set -x

pdm run ruff check siapy tests scripts --fix
pdm run ruff format siapy tests
