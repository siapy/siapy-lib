from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

if TYPE_CHECKING:
    from siapy.core.types import XarrayType

__all__ = [
    "ImageBase",
]


class ImageBase(ABC):
    """Abstract base class defining the interface for spectral image implementations.

    This class defines the common interface that all image backend implementations
    must implement, including methods for opening files, accessing metadata and
    properties, and converting to different formats.

    All concrete implementations (SpectralLibImage, RasterioLibImage, MockImage)
    must inherit from this class and implement all abstract methods.
    """

    @classmethod
    @abstractmethod
    def open(cls: type["ImageBase"], *args: Any, **kwargs: Any) -> "ImageBase":
        """Open and load an image from a source.

        Args:
            *args: Positional arguments specific to the implementation.
            **kwargs: Keyword arguments specific to the implementation.

        Returns:
            An instance of the concrete image implementation.

        Note:
            Each implementation defines its own signature for this method
            based on the specific requirements of the underlying library.
        """
        pass

    @property
    @abstractmethod
    def filepath(self) -> Path:
        """Get the file path of the image.

        Returns:
            A Path object representing the location of the image file. For in-memory images, this may return an empty Path.
        """
        pass

    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Get the image metadata.

        Returns:
            A dictionary containing image metadata such as coordinate reference system, geotransform information, wavelength data, and other image properties. The specific contents depend on the underlying format and library.
        """
        pass

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int, int]:
        """Get the dimensions of the image.

        Returns:
            A tuple (height, width, bands) representing the image dimensions.
        """
        pass

    @property
    @abstractmethod
    def bands(self) -> int:
        """Get the number of spectral bands in the image.

        Returns:
            The number of spectral bands (channels) in the image.
        """
        pass

    @property
    @abstractmethod
    def default_bands(self) -> list[int]:
        """Get the default band indices for RGB display.

        Returns:
            A list of band indices typically used for red, green, and blue channels when displaying the image as an RGB composite.
        """
        pass

    @property
    @abstractmethod
    def wavelengths(self) -> list[float]:
        """Get the wavelengths corresponding to each spectral band.

        Returns:
            A list of wavelength values (typically in nanometers) for each band. For non-spectral data, this may return band numbers or other identifiers.
        """
        pass

    @property
    @abstractmethod
    def camera_id(self) -> str:
        """Get the camera or sensor identifier.

        Returns:
            A string identifying the camera or sensor used to capture the image. May return an empty string if no camera information is available.
        """
        pass

    @abstractmethod
    def to_display(self, equalize: bool = True) -> Image.Image:
        """Convert the image to a PIL Image for display purposes.

        Args:
            equalize: Whether to apply histogram equalization to enhance contrast.

        Returns:
            A PIL Image object suitable for display, typically as an RGB composite created from the default bands with appropriate scaling and normalization.
        """
        pass

    @abstractmethod
    def to_numpy(self, nan_value: float | None = None) -> NDArray[np.floating[Any]]:
        """Convert the image to a numpy array.

        Args:
            nan_value: Optional value to replace NaN values with. If None, NaN values are preserved.

        Returns:
            A 3D numpy array with shape (height, width, bands) containing the image data. The array dtype should be a floating-point type.
        """
        pass

    @abstractmethod
    def to_xarray(self) -> "XarrayType":
        """Convert the image to an xarray DataArray.

        Returns:
            An xarray DataArray with labeled dimensions and coordinates, suitable for advanced analysis and visualization. The array should include appropriate coordinate information and metadata attributes.
        """
        pass
