from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from PIL import Image

from siapy.core.exceptions import InvalidInputError
from siapy.entities.images.interfaces import ImageBase

if TYPE_CHECKING:
    from siapy.core.types import XarrayType


class MockImage(ImageBase):
    def __init__(
        self,
        array: NDArray[np.floating[Any]],
    ) -> None:
        """Initialize a MockImage from a numpy array.

        Args:
            array: A 3D numpy array with shape (height, width, bands) containing spectral data.
                   The array will be automatically converted to float32 dtype.

        Raises:
            InvalidInputError: If the input array is not 3-dimensional.

        Example:
            ```python
            import numpy as np

            # Create a synthetic spectral image
            data = np.random.rand(100, 100, 10).astype(np.float32)
            mock_image = MockImage(data)
            ```

        Note:
            The input array is automatically converted to float32 dtype regardless of input type.
        """
        if len(array.shape) != 3:
            raise InvalidInputError(
                input_value=array.shape,
                message="Input array must be 3-dimensional (height, width, bands)",
            )

        self._array = array.astype(np.float32)

    @classmethod
    def open(cls, array: NDArray[np.floating[Any]]) -> "MockImage":
        """Create a MockImage instance from a numpy array.

        Args:
            array: A 3D numpy array with shape (height, width, bands) containing spectral data.

        Returns:
            A MockImage instance wrapping the provided array.

        Example:
            ```python
            import numpy as np

            # Create synthetic data
            data = np.random.rand(50, 50, 5).astype(np.float32)
            mock_image = MockImage.open(data)
            ```
        """
        return cls(array=array)

    @property
    def filepath(self) -> Path:
        """Get a placeholder file path for the mock image.

        Returns:
            An empty Path object since mock images are not associated with files.
        """
        return Path()

    @property
    def metadata(self) -> dict[str, Any]:
        """Get empty metadata for the mock image.

        Returns:
            An empty dictionary since mock images have no associated metadata.
        """
        return {}

    @property
    def shape(self) -> tuple[int, int, int]:
        """Get the dimensions of the mock image.

        Returns:
            A tuple (height, width, bands) representing the image dimensions extracted from the underlying numpy array shape.
        """
        x = self._array.shape[1]
        y = self._array.shape[0]
        bands = self._array.shape[2]
        return (y, x, bands)

    @property
    def bands(self) -> int:
        """Get the number of spectral bands in the mock image.

        Returns:
            The number of bands (third dimension) in the underlying array.
        """
        return self._array.shape[2]

    @property
    def default_bands(self) -> list[int]:
        """Get the default band indices for RGB display.

        Returns:
            A list of band indices for RGB display. If the image has 3 or more bands, returns [0, 1, 2]. Otherwise, returns indices for all available bands up to a maximum of 3.
        """
        if self.bands >= 3:
            return [0, 1, 2]
        return list(range(min(3, self.bands)))

    @property
    def wavelengths(self) -> list[float]:
        """Get placeholder wavelength values for the mock image.

        Returns:
            A list of sequential integers as float values, starting from 0, representing mock wavelengths for each band.
        """
        return list(range(self.bands))

    @property
    def camera_id(self) -> str:
        """Get a placeholder camera identifier for the mock image.

        Returns:
            An empty string since mock images are not associated with real cameras.
        """
        return ""

    def to_display(self, equalize: bool = True) -> Image.Image:
        """Convert the mock image to a PIL Image for display purposes.

        Args:
            equalize: Whether to apply histogram equalization to enhance contrast. Defaults to True.

        Returns:
            A PIL Image object suitable for display. For images with 3+ bands, uses the first 3 bands as RGB. For images with fewer bands, duplicates the first band across all RGB channels.

        Example:
            ```python
            # Display the mock image
            pil_image = mock_image.to_display()
            pil_image.show()

            # Display without histogram equalization
            pil_image = mock_image.to_display(equalize=False)
            ```

        Note:
            NaN values in the image are automatically handled by replacing them with 0
            during the scaling process. The method always returns an RGB image regardless
            of the number of input bands.
        """
        if self.bands >= 3:
            display_bands = self._array[:, :, self.default_bands]
        else:
            display_bands = np.stack([self._array[:, :, 0]] * 3, axis=2)

        if equalize:
            for i in range(display_bands.shape[2]):
                band = display_bands[:, :, i]
                non_nan = ~np.isnan(band)
                if np.any(non_nan):
                    min_val = np.nanmin(band)
                    max_val = np.nanmax(band)
                    if max_val > min_val:
                        band = (band - min_val) / (max_val - min_val) * 255
                    display_bands[:, :, i] = band

        display_array = np.nan_to_num(display_bands).astype(np.uint8)
        return Image.fromarray(display_array)

    def to_numpy(self, nan_value: float | None = None) -> NDArray[np.floating[Any]]:
        """Convert the mock image to a numpy array.

        Args:
            nan_value: Optional value to replace NaN values with. If None, NaN values are preserved.

        Returns:
            A copy of the underlying 3D numpy array with shape (height, width, bands). If nan_value is provided, all NaN values are replaced with this value.

        Example:
            ```python
            # Get the raw data with NaN values preserved
            data = mock_image.to_numpy()

            # Replace NaN values with zero
            data = mock_image.to_numpy(nan_value=0.0)
            ```
        """
        if nan_value is not None:
            return np.nan_to_num(self._array, nan=nan_value)
        return self._array.copy()

    def to_xarray(self) -> "XarrayType":
        """Convert the mock image to an xarray DataArray.

        Returns:
            An xarray DataArray with labeled dimensions (y, x, band) and coordinates, including mock wavelength values and minimal metadata.

        Example:
            ```python
            # Convert to xarray for analysis
            xr_data = mock_image.to_xarray()

            # Access specific bands
            first_band = xr_data.sel(band=0)
            ```
        """
        return xr.DataArray(
            self._array,
            dims=["y", "x", "band"],
            coords={
                "band": self.wavelengths,
                "x": np.arange(self.shape[1]),
                "y": np.arange(self.shape[0]),
            },
            attrs={
                "camera_id": self.camera_id,
            },
        )
