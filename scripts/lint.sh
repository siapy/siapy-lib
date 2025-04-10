#!/usr/bin/env bash

set -e
set -x

docs_src="docs/examples/src"

pdm run mypy siapy tests $docs_src
pdm run ruff check siapy tests scripts $docs_src
pdm run ruff format siapy tests $docs_src --check
