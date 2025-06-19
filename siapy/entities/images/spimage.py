from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Iterable, Sequence, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image

from ..pixels import CoordinateInput, Pixels, validate_pixel_input
from ..shapes import GeometricShapes, Shape
from ..signatures import Signatures
from .interfaces import ImageBase
from .mock import MockImage
from .rasterio_lib import RasterioLibImage
from .spectral_lib import SpectralLibImage

if TYPE_CHECKING:
    from siapy.core.types import XarrayType


__all__ = [
    "SpectralImage",
]

T = TypeVar("T", bound=ImageBase)


@dataclass
class SpectralImage(Generic[T]):
    def __init__(
        self,
        image: T,
        geometric_shapes: list["Shape"] | None = None,
    ):
        """Initialize a SpectralImage wrapper around an image backend.

        Args:
            image: The underlying image implementation (e.g., RasterioLibImage, SpectralLibImage, MockImage).
            geometric_shapes: Optional list of geometric shapes associated with this image. Defaults to None.
        """
        self._image = image
        self._geometric_shapes = GeometricShapes(self, geometric_shapes)

    def __repr__(self) -> str:
        """Return a string representation of the SpectralImage.

        Returns:
            A formatted string showing the SpectralImage and its underlying image.
        """
        return f"SpectralImage(\n{self.image}\n)"

    def __lt__(self, other: "SpectralImage[Any]") -> bool:
        """Compare two SpectralImage instances by their filepath names.

        Args:
            other: Another SpectralImage instance to compare against.

        Returns:
            True if this image's filepath name is lexicographically less than the other's.
        """
        return self.filepath.name < other.filepath.name

    def __eq__(self, other: Any) -> bool:
        """Check equality between two SpectralImage instances.

        Args:
            other: Another object to compare against.

        Returns:
            True if both images have the same filepath name and underlying image implementation. NotImplemented if the other object is not a SpectralImage.
        """
        if not isinstance(other, SpectralImage):
            return NotImplemented
        return self.filepath.name == other.filepath.name and self._image == other._image

    def __array__(self, dtype: np.dtype | None = None) -> NDArray[np.floating[Any]]:
        """Convert this spectral image to a numpy array when requested by NumPy.

        This method enables the SpectralImage to be used directly with NumPy functions
        that expect array-like objects.

        Args:
            dtype: Optional numpy data type to cast the array to. Defaults to None.

        Returns:
            A numpy array representation of the spectral image data.
        """
        array = self.to_numpy()
        if dtype is not None:
            return array.astype(dtype)
        return array

    @classmethod
    def spy_open(
        cls, *, header_path: str | Path, image_path: str | Path | None = None
    ) -> "SpectralImage[SpectralLibImage]":
        """Open a spectral image using the SpectralPython library backend.

        Args:
            header_path: Path to the header file (.hdr) containing image metadata.
            image_path: Optional path to the image data file. If None, inferred from header_path.

        Returns:
            A SpectralImage instance wrapping a SpectralLibImage backend.

        Example:
            ```python
            # Open an ENVI format image
            image = SpectralImage.spy_open(header_path="image.hdr")

            # Open with explicit image file path
            image = SpectralImage.spy_open(
                header_path="image.hdr",
                image_path="image.dat"
            )
            ```
        """
        image = SpectralLibImage.open(header_path=header_path, image_path=image_path)
        return SpectralImage(image)

    @classmethod
    def rasterio_open(cls, filepath: str | Path) -> "SpectralImage[RasterioLibImage]":
        """Open a spectral image using the Rasterio library backend.

        Args:
            filepath: Path to the image file (supports formats like GeoTIFF, etc.).

        Returns:
            A SpectralImage instance wrapping a RasterioLibImage backend.

        Example:
            ```python
            # Open a GeoTIFF file
            image = SpectralImage.rasterio_open("image.tif")
            ```
        """
        image = RasterioLibImage.open(filepath)
        return SpectralImage(image)

    @classmethod
    def from_numpy(cls, array: NDArray[np.floating[Any]]) -> "SpectralImage[MockImage]":
        """Create a spectral image from a numpy array using the mock backend.

        Args:
            array: A 3D numpy array with shape (height, width, bands) containing spectral data.

        Returns:
            A SpectralImage instance wrapping a MockImage backend.

        Example:
            ```python
            import numpy as np

            # Create a synthetic spectral image
            data = np.random.rand(100, 100, 10)  # 100x100 image with 10 bands
            image = SpectralImage.from_numpy(data)
            ```
        """
        image = MockImage.open(array)
        return SpectralImage(image)

    @property
    def image(self) -> T:
        """Get the underlying image implementation.

        Returns:
            The wrapped image backend instance (e.g., RasterioLibImage, SpectralLibImage, MockImage).
        """
        return self._image

    @property
    def geometric_shapes(self) -> GeometricShapes:
        """Get the geometric shapes associated with this image.

        Returns:
            A GeometricShapes instance containing all shapes linked to this image.
        """
        return self._geometric_shapes

    @property
    def filepath(self) -> Path:
        """Get the file path of the image.

        Returns:
            A Path object representing the location of the image file.
        """
        return self.image.filepath

    @property
    def metadata(self) -> dict[str, Any]:
        """Get the image metadata.

        Returns:
            A dictionary containing image metadata such as coordinate reference system, geotransform information, and other image properties.
        """
        return self.image.metadata

    @property
    def shape(self) -> tuple[int, int, int]:
        """Get the dimensions of the image.

        Returns:
            A tuple (height, width, bands) representing the image dimensions.
        """
        return self.image.shape

    @property
    def width(self) -> int:
        """Get the width of the image in pixels.

        Returns:
            The number of pixels in the horizontal dimension.
        """
        return self.shape[1]

    @property
    def height(self) -> int:
        """Get the height of the image in pixels.

        Returns:
            The number of pixels in the vertical dimension.
        """
        return self.shape[0]

    @property
    def bands(self) -> int:
        """Get the number of spectral bands in the image.

        Returns:
            The number of spectral bands (channels) in the image.
        """
        return self.image.bands

    @property
    def default_bands(self) -> list[int]:
        """Get the default band indices for RGB display.

        Returns:
            A list of band indices typically used for red, green, and blue channels when displaying the image as an RGB composite.
        """
        return self.image.default_bands

    @property
    def wavelengths(self) -> list[float]:
        """Get the wavelengths corresponding to each spectral band.

        Returns:
            A list of wavelength values (typically in nanometers) for each band. The length of this list equals the number of bands.
        """
        return self.image.wavelengths

    @property
    def camera_id(self) -> str:
        """Get the camera or sensor identifier.

        Returns:
            A string identifying the camera or sensor used to capture the image.
        """
        return self.image.camera_id

    def to_display(self, equalize: bool = True) -> Image.Image:
        """Convert the image to a PIL Image for display purposes.

        Args:
            equalize: Whether to apply histogram equalization to enhance contrast. Defaults to True.

        Returns:
            A PIL Image object suitable for display, typically as an RGB composite.

        Example:
            ```python
            # Display the image with default settings
            pil_image = spectral_image.to_display()
            pil_image.show()

            # Display without histogram equalization
            pil_image = spectral_image.to_display(equalize=False)
            ```
        """
        return self.image.to_display(equalize)

    def to_numpy(self, nan_value: float | None = None) -> NDArray[np.floating[Any]]:
        """Convert the image to a numpy array.

        Args:
            nan_value: Optional value to replace NaN values with. If None, NaN values are preserved.

        Returns:
            A 3D numpy array with shape (height, width, bands) containing the spectral data.

        Example:
            ```python
            # Get the raw data with NaN values preserved
            data = spectral_image.to_numpy()

            # Replace NaN values with zero
            data = spectral_image.to_numpy(nan_value=0.0)
            ```
        """
        return self.image.to_numpy(nan_value)

    def to_xarray(self) -> "XarrayType":
        """Convert the image to an xarray DataArray.

        Returns:
            An xarray DataArray with labeled dimensions and coordinates, suitable for advanced analysis and visualization.

        Example:
            ```python
            # Convert to xarray for analysis
            xr_data = spectral_image.to_xarray()

            # Access specific bands or wavelengths
            red_band = xr_data.sel(wavelength=650, method='nearest')
            ```
        """
        return self.image.to_xarray()

    def to_signatures(self, pixels: Pixels | pd.DataFrame | Iterable[CoordinateInput]) -> Signatures:
        """Extract spectral signatures from specific pixel locations.

        Args:
            pixels: Pixel coordinates to extract signatures from. Can be a Pixels object,
                    pandas DataFrame with 'x' and 'y' columns, or an iterable of coordinate tuples.

        Returns:
            A Signatures object containing the spectral data for the specified pixels.

        Example:
            ```python
            # Extract signatures from specific coordinates
            coords = [(10, 20), (30, 40), (50, 60)]
            signatures = spectral_image.to_signatures(coords)

            # Extract signatures from a DataFrame
            import pandas as pd
            df = pd.DataFrame({'x': [10, 30, 50], 'y': [20, 40, 60]})
            signatures = spectral_image.to_signatures(df)
            ```
        """
        pixels = validate_pixel_input(pixels)
        image_arr = self.to_numpy()
        signatures = Signatures.from_array_and_pixels(image_arr, pixels)
        return signatures

    def to_subarray(self, pixels: Pixels | pd.DataFrame | Iterable[CoordinateInput]) -> NDArray[np.floating[Any]]:
        """Extract a rectangular subarray containing the specified pixels.

        Creates a new array that encompasses all the specified pixel coordinates,
        with NaN values for pixels not in the original selection.

        Args:
            pixels: Pixel coordinates defining the region of interest. Can be a Pixels object,
                    pandas DataFrame with 'x' and 'y' columns, or an iterable of coordinate tuples.

        Returns:
            A 3D numpy array containing the subregion with shape (height, width, bands). Unselected pixels within the bounding rectangle are filled with NaN.

        Example:
            ```python
            # Extract a subarray around specific points
            coords = [(10, 20), (15, 25), (12, 22)]
            subarray = spectral_image.to_subarray(coords)

            # The resulting array will be 6x6x{bands} covering the bounding box
            # from (10,20) to (15,25) with only the specified pixels having data
            ```
        """
        pixels = validate_pixel_input(pixels)
        image_arr = self.to_numpy()
        x_max = pixels.x().max()
        x_min = pixels.x().min()
        y_max = pixels.y().max()
        y_min = pixels.y().min()
        # create new image
        image_arr_area = np.nan * np.ones((int(y_max - y_min + 1), int(x_max - x_min + 1), self.bands))
        # convert original coordinates to coordinates for new image
        y_norm = pixels.y() - y_min
        x_norm = pixels.x() - x_min
        # write values from original image to new image
        image_arr_area[y_norm, x_norm, :] = image_arr[pixels.y(), pixels.x(), :]
        return image_arr_area

    def average_intensity(
        self, axis: int | tuple[int, ...] | Sequence[int] | None = None
    ) -> float | NDArray[np.floating[Any]]:
        """Calculate the average intensity across specified dimensions.

        Args:
            axis: Axis or axes along which to compute the mean. If None, computes the mean over the entire array. Can be an integer, tuple of integers, or sequence of integers. <br>
                  - axis=None: Average over all pixels and bands (returns single float) <br>
                  - axis=(0,1): Average over spatial dimensions (returns array of band averages) <br>
                  - axis=2: Average over spectral dimension (returns spatial average image)

        Returns:
            Either a single float (if axis=None) or a numpy array with reduced dimensions. NaN values are ignored in the calculation.

        Example:
            ```python
            # Get overall average intensity
            overall_avg = spectral_image.average_intensity()

            # Get average spectrum (average over spatial dimensions)
            avg_spectrum = spectral_image.average_intensity(axis=(0, 1))

            # Get spatial average (average over spectral dimension)
            spatial_avg = spectral_image.average_intensity(axis=2)
            ```
        """
        image_arr = self.to_numpy()
        return np.nanmean(image_arr, axis=axis)
