# mypy: ignore-errors
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import spectral as sp
import xarray as xr
from numpy.typing import NDArray
from PIL import Image, ImageOps

from siapy.core.exceptions import InvalidFilepathError, InvalidInputError

from .interfaces import ImageBase

if TYPE_CHECKING:
    from siapy.core.types import SpectralLibType, XarrayType

__all__ = [
    "SpectralLibImage",
]


@dataclass
class SpectralLibImage(ImageBase):
    def __init__(
        self,
        file: "SpectralLibType",
    ):
        """Initialize a SpectralLibImage wrapper around a SpectralPython file object.

        Args:
            file: A SpectralPython file object representing the opened spectral image.
        """
        self._file = file

    @classmethod
    def open(cls, *, header_path: str | Path, image_path: str | Path | None = None) -> "SpectralLibImage":
        """Open a spectral image using the SpectralPython ENVI library.

        Args:
            header_path: Path to the ENVI header file (.hdr) containing image metadata.
            image_path: Optional path to the image data file. If None, the path is inferred from the header file.

        Returns:
            A SpectralLibImage instance wrapping the opened spectral file.

        Raises:
            InvalidFilepathError: If the header file path does not exist.
            InvalidInputError: If the file cannot be opened or if it's a SpectralLibrary instead of an Image.

        Example:
            ```python
            # Open an ENVI format image
            image = SpectralLibImage.open(header_path="image.hdr")

            # Open with explicit image file path
            image = SpectralLibImage.open(
                header_path="image.hdr",
                image_path="image.dat"
            )
            ```
        """
        header_path = Path(header_path)
        if not header_path.exists():
            raise InvalidFilepathError(header_path)

        try:
            sp_file = sp.envi.open(file=header_path, image=image_path)
        except Exception as e:
            raise InvalidInputError({"filepath": str(header_path)}, f"Failed to open spectral file: {e}") from e

        if isinstance(sp_file, sp.io.envi.SpectralLibrary):
            raise InvalidInputError({"file_type": type(sp_file).__name__}, "Expected Image, got SpectralLibrary")

        return cls(sp_file)

    @property
    def file(self) -> "SpectralLibType":
        """Get the underlying SpectralPython file object.

        Returns:
            The wrapped SpectralPython file object.
        """
        return self._file

    @property
    def filepath(self) -> Path:
        """Get the file path of the spectral image.

        Returns:
            A Path object representing the location of the image file.
        """
        return Path(self.file.filename)

    @property
    def metadata(self) -> dict[str, Any]:
        """Get the image metadata from the ENVI header.

        Returns:
            A dictionary containing image metadata such as coordinate reference system, wavelength information, and other image properties from the ENVI header.
        """
        return self.file.metadata

    @property
    def shape(self) -> tuple[int, int, int]:
        """Get the dimensions of the spectral image.

        Returns:
            A tuple (rows, samples, bands) representing the image dimensions.
        """
        rows = self.file.nrows
        samples = self.file.ncols
        bands = self.file.nbands
        return (rows, samples, bands)

    @property
    def rows(self) -> int:
        """Get the number of rows (height) in the image.

        Returns:
            The number of rows in the spectral image.
        """
        return self.file.nrows

    @property
    def cols(self) -> int:
        """Get the number of columns (width) in the image.

        Returns:
            The number of columns in the spectral image.
        """
        return self.file.ncols

    @property
    def bands(self) -> int:
        """Get the number of spectral bands in the image.

        Returns:
            The number of spectral bands (channels) in the image.
        """
        return self.file.nbands

    @property
    def default_bands(self) -> list[int]:
        """Get the default band indices for RGB display.

        Returns:
            A list of 0-based band indices from the ENVI metadata, typically used for red, green, and blue channels when displaying the image as an RGB composite. These indices are extracted from the "default bands" field in the ENVI header file.

        Note:
            Unlike rasterio which uses 1-based indexing, SpectralPython uses 0-based indexing
            for band access. The returned indices can be used directly with numpy array indexing.
        """
        db = self.metadata.get("default bands", [])
        return list(map(int, db))

    @property
    def wavelengths(self) -> list[float]:
        """Get the wavelengths corresponding to each spectral band.

        Returns:
            A list of wavelength values (typically in nanometers) for each band as stored in the ENVI metadata. The length equals the number of bands.
        """
        wavelength_data = self.metadata.get("wavelength", [])
        return list(map(float, wavelength_data))

    @property
    def description(self) -> dict[str, Any]:
        """Get parsed description metadata from the ENVI header.

        Returns:
            A dictionary containing parsed key-value pairs from the description field in the ENVI metadata, with automatic type conversion for numeric values and comma-separated lists.
        """
        description_str = self.metadata.get("description", {})
        return _parse_description(description_str)

    @property
    def camera_id(self) -> str:
        """Get the camera or sensor identifier from the description metadata.

        Returns:
            A string identifying the camera or sensor used to capture the image, extracted from the "ID" field in the parsed description metadata.
        """
        return self.description.get("ID", "")

    def to_display(self, equalize: bool = True) -> Image.Image:
        """Convert the image to a PIL Image for display purposes.

        Args:
            equalize: Whether to apply histogram equalization to enhance contrast. Defaults to True.

        Returns:
            A PIL Image object suitable for display, created from the default RGB bands with normalized values and optional histogram equalization.

        Example:
            ```python
            # Display the image with default settings
            pil_image = spectral_image.to_display()
            pil_image.show()

            # Display without histogram equalization
            pil_image = spectral_image.to_display(equalize=False)
            ```
        """
        max_uint8 = 255.0
        image_3ch = self.file.read_bands(self.default_bands)
        image_3ch = self._remove_nan(image_3ch, nan_value=0)
        image_3ch[:, :, 0] = image_3ch[:, :, 0] / image_3ch[:, :, 0].max() / max_uint8
        image_3ch[:, :, 1] = image_3ch[:, :, 1] / (image_3ch[:, :, 1].max() / max_uint8)
        image_3ch[:, :, 2] = image_3ch[:, :, 2] / (image_3ch[:, :, 2].max() / max_uint8)
        image = Image.fromarray(image_3ch.astype("uint8"))
        if equalize:
            image = ImageOps.equalize(image)
        return image

    def to_numpy(self, nan_value: float | None = None) -> NDArray[np.floating[Any]]:
        """Convert the image to a numpy array.

        Args:
            nan_value: Optional value to replace NaN values with. If None, NaN values are preserved.

        Returns:
            A 3D numpy array with shape (rows, cols, bands) containing the spectral data.

        Example:
            ```python
            # Get the raw data with NaN values preserved
            data = spectral_image.to_numpy()

            # Replace NaN values with zero
            data = spectral_image.to_numpy(nan_value=0.0)
            ```
        """
        image = self.file[:, :, :]
        if nan_value is not None:
            image = self._remove_nan(image, nan_value)
        return image

    def _remove_nan(self, image: np.ndarray, nan_value: float = 0.0) -> np.ndarray:
        """Replace NaN values in the image array with a specified value.

        Args:
            image: The input image array that may contain NaN values.
            nan_value: The value to replace NaN values with. Defaults to 0.0.

        Returns:
            The image array with NaN values replaced by the specified value.

        Note:
            This method identifies pixels that have NaN values in any band and replaces
            ALL values for those pixels (across all bands) with the specified nan_value.
            This ensures that pixels are either completely valid or completely invalid.

        Example:
            If a pixel at position (0,0) has NaN in any band, all bands for that pixel
            will be set to nan_value, not just the bands containing NaN.
        """
        image_mask = np.bitwise_not(np.bool_(np.isnan(image).sum(axis=2)))
        image[~image_mask] = nan_value
        return image

    def to_xarray(self) -> "XarrayType":
        """Convert the image to an xarray DataArray.

        Returns:
            An xarray DataArray with labeled dimensions (y, x, band) and coordinates, including wavelength information and metadata attributes.

        Example:
            ```python
            # Convert to xarray for analysis
            xr_data = spectral_image.to_xarray()

            # Access specific bands or wavelengths
            red_band = xr_data.sel(band=650, method='nearest')
            ```
        """
        data = self._file[:, :, :]
        xarray = xr.DataArray(
            data,
            dims=["y", "x", "band"],
            coords={
                "y": np.arange(self.rows),
                "x": np.arange(self.cols),
                "band": self.wavelengths,
            },
            attrs=self._file.metadata,
        )
        return xarray


def _parse_description(description: str) -> dict[str, Any]:
    """Parse the description string from ENVI metadata into a structured dictionary.

    Parses key-value pairs from a description string, where each line contains
    a key=value pair. Automatically converts numeric values and comma-separated lists.

    Args:
        description: The description string from ENVI metadata containing key=value pairs.
                    Each line should be in the format "key=value".

    Returns:
        A dictionary with parsed key-value pairs, with automatic type conversion:
        - Integer values are converted to int
        - Float values are converted to float
        - Comma-separated values are converted to lists with type conversion applied to each element
        - Other values remain as strings
        - Empty values become empty strings

    Raises:
        InvalidInputError: If the description cannot be parsed due to malformed format,
                          missing keys, or other parsing errors. This includes cases where
                          lines don't contain '=' separators.

    Example:
        ```python
        desc = "ID=Camera1\nExposure=100\nWavelengths=400.0,500.0,600.0\nComment="
        result = _parse_description(desc)
        # Returns: {
        #     "ID": "Camera1",
        #     "Exposure": 100,
        #     "Wavelengths": [400.0, 500.0, 600.0],
        #     "Comment": ""
        # }
        ```

    Note:
        The function expects each line to contain exactly one '=' character separating
        the key and value. Lines without '=' will cause a ValueError which is caught
        and re-raised as InvalidInputError.
    """

    def _parse():
        data_dict = {}
        for line in description.split("\n"):
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if "," in value:  # Special handling for values with commas
                value = [float(v) if v.replace(".", "", 1).isdigit() else v for v in value.split(",")]
            elif value.isdigit():
                value = int(value)
            elif value.replace(".", "", 1).isdigit():
                value = float(value)
            data_dict[key] = value
        return data_dict

    try:
        return _parse()

    except ValueError as e:
        raise InvalidInputError(
            {
                "description": description,
                "error": str(e),
            },
            f"Error parsing description: {e}",
        ) from e
    except KeyError as e:
        raise InvalidInputError(
            {
                "description": description,
                "error": str(e),
            },
            f"Missing key in description: {e}",
        ) from e
    except Exception as e:
        raise InvalidInputError(
            {
                "description": description,
                "error": str(e),
            },
            f"Unexpected error parsing description: {e}",
        ) from e
