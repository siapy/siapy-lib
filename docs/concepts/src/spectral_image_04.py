from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from PIL import Image

from siapy.core import logger
from siapy.core.exceptions import InvalidFilepathError
from siapy.entities import SpectralImage
from siapy.entities.images import ImageBase

if TYPE_CHECKING:
    from siapy.core.types import XarrayType


class MyImage(ImageBase):
    """
    # Create your own image class by extending ImageBase
    # This example demonstrates how to implement a custom image loader
    """

    def __init__(self, data: NDArray[np.floating[Any]], file_path: Path) -> None:
        self._data = data
        self._filepath = file_path

        # Define metadata with required fields:
        # - camera_id: unique identifier for the imaging device
        # - wavelengths: list of spectral band centers in nanometers
        # - default_bands: which bands to use for RGB visualization
        self._meta: dict[str, Any] = {
            "camera_id": "my_camera",
            "wavelengths": [450.0, 550.0, 650.0],  # RGB wavelengths in nm
            "default_bands": [0, 1, 2],  # Band indices for RGB display
        }

    @classmethod
    def open(cls, filepath: str) -> "MyImage":
        """Load an image from a file path"""
        path = Path(filepath)
        if not path.exists():
            raise InvalidFilepathError(f"File not found: {filepath}")

        try:
            # This is a simplified example - in a real implementation,
            # you would read the actual image data using an appropriate library
            # For example purposes, creating a small random array
            data = np.random.random((100, 100, 3)).astype(np.float32)
            return cls(data, path)
        except Exception as e:
            raise InvalidFilepathError(f"Failed to open {filepath}: {str(e)}")

    # Required properties (all must be implemented)

    @property
    def filepath(self) -> Path:
        """Path to the source file"""
        return self._filepath

    @property
    def metadata(self) -> dict[str, Any]:
        """Image metadata dictionary"""
        return self._meta

    @property
    def shape(self) -> tuple[int, int, int]:
        """Image dimensions as (height, width, bands)"""
        return cast(tuple[int, int, int], self._data.shape)

    @property
    def bands(self) -> int:
        """Number of spectral bands"""
        return self.shape[2]

    @property
    def default_bands(self) -> list[int]:
        """Indices of bands to use for RGB visualization"""
        return self._meta["default_bands"]

    @property
    def wavelengths(self) -> list[float]:
        """Center wavelengths of each band in nanometers"""
        return self._meta["wavelengths"]

    @property
    def camera_id(self) -> str:
        """Unique identifier for the imaging device"""
        return self._meta["camera_id"]

    # Required methods (all must be implemented)

    def to_display(self, equalize: bool = True) -> Image.Image:
        """Convert to PIL Image for display"""
        # Extract the default bands for RGB visualization
        rgb_data = self._data[:, :, self.default_bands].copy()

        if equalize:
            # Apply linear contrast stretching to each band
            for i in range(rgb_data.shape[2]):
                band = rgb_data[:, :, i]
                min_val = np.min(band)
                max_val = np.max(band)
                if max_val > min_val:
                    rgb_data[:, :, i] = (band - min_val) / (max_val - min_val)

        # Convert to 8-bit for PIL
        rgb_uint8 = (rgb_data * 255).astype(np.uint8)
        return Image.fromarray(rgb_uint8)

    def to_numpy(self, nan_value: float | None = None) -> NDArray[np.floating[Any]]:
        """Convert to numpy array"""
        result = self._data.copy()
        if nan_value is not None:
            result[np.isnan(result)] = nan_value
        return result

    def to_xarray(self) -> "XarrayType":
        """Convert to xarray DataArray with coordinates"""
        return xr.DataArray(
            self._data,
            dims=["y", "x", "band"],
            coords={
                "y": np.arange(self.shape[0]),
                "x": np.arange(self.shape[1]),
                "band": self.wavelengths,
            },
            attrs=self.metadata,
        )


# Example: Using your custom image class with SiaPy
# 1. Create an instance of your custom image class
custom_image = MyImage.open("path/to/your/image.dat")

# 2. Wrap it in a SpectralImage for use with SiaPy's analysis tools
spectral_image = SpectralImage(custom_image)

# 3. Now you can use all SiaPy functionality
# spectral_image.to_signatures(pixels)
# etc.
