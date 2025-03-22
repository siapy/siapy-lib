#!/usr/bin/env bash

set -e

# Check if a version number is provided and validate that it is an integer
if [ -z "$1" ]; then
  echo "Usage: $0 <version number>"
  exit 1
fi

version="$1"
if ! [[ "$version" =~ ^[0-9]+$ ]]; then
  echo "Error: Version must be an integer."
  exit 1
fi

set -x

# Get the directory where the script is located and move there
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tests_dir="$script_dir/../tests"
cd "$tests_dir"

# Create a compressed archive using the provided version
tar -czvf "testdata-v${version}.tar.gz" --exclude="*.tar.gz*" data/
mv "testdata-v${version}.tar.gz" "data/testdata-v${version}.tar.gz"

# Add a checksum file to verify integrity
sha256sum "data/testdata-v${version}.tar.gz" >"data/testdata-v${version}.tar.gz.sha256"

## -- Uncomment the following lines to push the tag to the remote repository --
# # Push tag to remote repository
# git tag "testdata-v${version}"
# git push origin "testdata-v${version}"

set +x
echo "Archive created at: $tests_dir/data/testdata-v${version}.tar.gz"
echo "Checksum file created at: $tests_dir/data/testdata-v${version}.tar.gz.sha256"
