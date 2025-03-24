#!/usr/bin/env bash

set -e
set -x

# Install pdm
curl -sSL https://pdm-project.org/install-pdm.py | python3 -

# Install python libraries
pdm install

# Install pre-commit
pdm run pre-commit uninstall
pdm run pre-commit install # --hook-type commit-msg
