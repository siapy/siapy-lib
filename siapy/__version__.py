from importlib import metadata

try:
    __version__ = metadata.version("siapy")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"
