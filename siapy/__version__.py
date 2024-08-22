import os
import re

# Define the path to the pyproject.toml file
pyproject_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "pyproject.toml"
)

# Read the pyproject.toml file
with open(pyproject_path, "r", encoding="utf-8") as f:
    pyproject_content = f.read()

# Use a regular expression to extract the version number
version_match = re.search(r'version\s*=\s*"([^"]+)"', pyproject_content)
if version_match:
    __version__ = version_match.group(1)
else:
    raise ValueError("Version number not found in pyproject.toml")
