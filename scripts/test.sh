#!/usr/bin/env bash

set -e
set -x

pdm run coverage run --source=siapy -m pytest -m "not manual"
pdm run coverage report --show-missing
pdm run coverage html --title "${@-coverage}"
