from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import rioxarray
from numpy.typing import NDArray
from PIL import Image, ImageOps

from siapy.core.exceptions import InvalidFilepathError, InvalidInputError

from .interfaces import ImageBase

if TYPE_CHECKING:
    from siapy.core.types import XarrayType

__all__ = [
    "RasterioLibImage",
]


@dataclass
class RasterioLibImage(ImageBase):
    def __init__(self, file: "XarrayType"):
        """Initialize a RasterioLibImage wrapper around an xarray DataArray.

        Args:
            file: An xarray DataArray containing the raster data loaded via rioxarray.
        """
        self._file = file

    @classmethod
    def open(cls, filepath: str | Path) -> "RasterioLibImage":
        """Open a raster image using the rioxarray library.

        Args:
            filepath: Path to the raster file (supports formats like GeoTIFF, NetCDF, HDF5, etc.).

        Returns:
            A RasterioLibImage instance wrapping the opened raster data.

        Raises:
            InvalidFilepathError: If the file path does not exist.
            InvalidInputError: If the file cannot be opened (e.g., unsupported format,
                             corrupted file) or if rioxarray returns a list instead of
                             DataArray/Dataset.

        Example:
            ```python
            # Open a GeoTIFF file
            image = RasterioLibImage.open("satellite_image.tif")

            # Open a NetCDF file
            image = RasterioLibImage.open("climate_data.nc")
            ```

        Note:
            The method uses rioxarray.open_rasterio() internally, which supports most
            GDAL-compatible raster formats. Some formats may require additional dependencies.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise InvalidFilepathError(filepath)

        try:
            raster = rioxarray.open_rasterio(filepath)
        except Exception as e:
            raise InvalidInputError({"filepath": str(filepath)}, f"Failed to open raster file: {e}") from e

        if isinstance(raster, list):
            raise InvalidInputError({"file_type": type(raster).__name__}, "Expected DataArray or Dataset, got list")

        return cls(raster)

    @property
    def file(self) -> "XarrayType":
        """Get the underlying xarray DataArray.

        Returns:
            The wrapped xarray DataArray containing the raster data.
        """
        return self._file

    @property
    def filepath(self) -> Path:
        """Get the file path of the raster image.

        Returns:
            A Path object representing the location of the raster file, extracted from the xarray encoding information.
        """
        return Path(self.file.encoding["source"])

    @property
    def metadata(self) -> dict[str, Any]:
        """Get the raster metadata from the xarray attributes.

        Returns:
            A dictionary containing raster metadata such as coordinate reference system, geotransform information, and other properties stored in xarray attrs.
        """
        return self.file.attrs

    @property
    def shape(self) -> tuple[int, int, int]:
        """Get the dimensions of the raster image.

        Returns:
            A tuple (height, width, bands) representing the image dimensions. Note that rioxarray uses (band, y, x) ordering internally, which is converted to the standard (y, x, band) format.
        """
        # rioxarray uses (band, y, x) ordering
        return (self.file.y.size, self.file.x.size, self.file.band.size)

    @property
    def rows(self) -> int:
        """Get the number of rows (height) in the image.

        Returns:
            The number of rows in the raster image.
        """
        return self.file.y.size

    @property
    def cols(self) -> int:
        """Get the number of columns (width) in the image.

        Returns:
            The number of columns in the raster image.
        """
        return self.file.x.size

    @property
    def bands(self) -> int:
        """Get the number of bands in the raster image.

        Returns:
            The number of bands (channels) in the raster image.
        """
        return self.file.band.size

    @property
    def default_bands(self) -> list[int]:
        """Get the default band indices for RGB display.

        Returns:
            A list of 1-based band indices commonly used for RGB display. For satellite imagery, this typically returns [1, 2, 3] for images with 3+ bands, or [1, 2] for images with 2 bands, etc. Note that these are 1-based indices as used by rioxarray.

        Note:
            Returns 1-based band indices (not 0-based) to match rioxarray's band coordinate system.
            This differs from numpy array indexing which is 0-based.
        """
        # Most common RGB band combination for satellite imagery
        return list(range(1, min(3, self.bands) + 1))

    @property
    def wavelengths(self) -> list[float]:
        """Get the band values, which may represent wavelengths or band numbers.

        Returns:
            A numpy array of band coordinate values from the xarray DataArray. For spectral data, these may represent wavelengths in nanometers; for other raster data, these are typically just sequential band numbers (1, 2, 3, etc.).

        Note:
            The interpretation of these values depends on the source data. Check the metadata
            to determine if these represent actual wavelengths or just band identifiers.
        """
        return self.file.band.values

    @property
    def camera_id(self) -> str:
        """Get the camera or sensor identifier from metadata.

        Returns:
            A string identifying the camera or sensor used to capture the image. Returns empty string if no camera_id is found in the metadata.

        Note:
            This property looks for 'camera_id' in the metadata, which is not
            a standard raster metadata field and may not be present in most files.
        """
        return self.metadata.get("camera_id", "")

    def to_display(self, equalize: bool = True) -> Image.Image:
        """Convert the image to a PIL Image for display purposes.

        Args:
            equalize: Whether to apply histogram equalization to enhance contrast. Defaults to True.

        Returns:
            A PIL Image object suitable for display, created from the default RGB bands with values scaled to 0-255 range and optional histogram equalization.

        Example:
            ```python
            # Display the image with default settings
            pil_image = raster_image.to_display()
            pil_image.show()

            # Display without histogram equalization
            pil_image = raster_image.to_display(equalize=False)
            ```
        """
        bands_data = self.file.sel(band=self.default_bands)
        image_3ch = bands_data.transpose("y", "x", "band").values
        image_3ch_clean = np.nan_to_num(np.asarray(image_3ch))
        min_val = np.nanmin(image_3ch_clean)
        max_val = np.nanmax(image_3ch_clean)

        image_scaled = ((image_3ch_clean - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8)

        image = Image.fromarray(image_scaled)
        if equalize:
            image = ImageOps.equalize(image)
        return image

    def to_numpy(self, nan_value: float | None = None) -> NDArray[np.floating[Any]]:
        """Convert the image to a numpy array.

        Args:
            nan_value: Optional value to replace NaN values with. If None, NaN values are preserved.

        Returns:
            A 3D numpy array with shape (height, width, bands) containing the raster data. The array is transposed from rioxarray's native (band, y, x) to (y, x, band) format.

        Example:
            ```python
            # Get the raw data with NaN values preserved
            data = raster_image.to_numpy()

            # Replace NaN values with zero
            data = raster_image.to_numpy(nan_value=0.0)
            ```
        """
        image = np.asarray(self.file.transpose("y", "x", "band").values)
        if nan_value is not None:
            image = np.nan_to_num(image, nan=nan_value)
        return image

    def to_xarray(self) -> "XarrayType":
        """Convert the image to an xarray DataArray.

        Returns:
            The underlying xarray DataArray with all original coordinates, dimensions, and metadata preserved.

        Example:
            ```python
            # Access the xarray representation
            xr_data = raster_image.to_xarray()

            # Use xarray functionality
            subset = xr_data.sel(x=slice(100, 200), y=slice(100, 200))
            ```
        """
        return self.file
