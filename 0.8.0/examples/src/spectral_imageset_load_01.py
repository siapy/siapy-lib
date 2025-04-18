try:
    from pathlib import Path

    from siapy.entities import SpectralImageSet

    print("Libraries detected successfully.")
except ImportError as e:
    print(f"Error: {e}. Please ensure that the SiaPy library is installed and the environment is activated.")
    exit(1)

# Set the path to the directory containing the data
# !! ADJUST THIS PATH TO YOUR DATA DIRECTORY !!
data_dir = "./docs/examples/data"

# Find all header and image files in the data directory
header_paths = sorted(Path(data_dir).rglob("*.hdr"))
image_paths = sorted(Path(data_dir).rglob("*.img"))

# Create a SpectralImageSet from the found paths
image_set = SpectralImageSet.spy_open(
    header_paths=header_paths,
    image_paths=image_paths,
)

# Check if the data was loaded correctly
if len(image_set) > 0:
    print("Loading succeeded.")
else:
    print("Loading did not succeed.")
