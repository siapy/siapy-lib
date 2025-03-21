#!/usr/bin/env bash

set -e
set -x

# Get the directory where the script is located and move there
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tests_dir="$script_dir/../tests"
cd "$tests_dir"

# Create a compressed archive
tar -czvf testdata-v1.tar.gz data/
mv testdata-v1.tar.gz data/testdata-v1.tar.gz

# Add a checksum file to verify integrity
sha256sum data/testdata-v1.tar.gz >data/testdata-v1.tar.gz.sha256

set +x
echo "Archive created at: $tests_dir/data/testdata-v1.tar.gz"
echo "Checksum file created at: $tests_dir/data/testdata-v1.tar.gz.sha256"
