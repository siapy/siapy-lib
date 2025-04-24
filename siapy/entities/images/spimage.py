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
        self._image = image
        self._geometric_shapes = GeometricShapes(self, geometric_shapes)

    def __repr__(self) -> str:
        return f"SpectralImage(\n{self.image}\n)"

    def __lt__(self, other: "SpectralImage[Any]") -> bool:
        return self.filepath.name < other.filepath.name

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SpectralImage):
            return NotImplemented
        return self.filepath.name == other.filepath.name and self._image == other._image

    def __array__(self, dtype: np.dtype | None = None) -> NDArray[np.floating[Any]]:
        """Convert this spectral image to a numpy array when requested by NumPy."""
        array = self.to_numpy()
        if dtype is not None:
            return array.astype(dtype)
        return array

    @classmethod
    def spy_open(
        cls, *, header_path: str | Path, image_path: str | Path | None = None
    ) -> "SpectralImage[SpectralLibImage]":
        image = SpectralLibImage.open(header_path=header_path, image_path=image_path)
        return SpectralImage(image)

    @classmethod
    def rasterio_open(cls, filepath: str | Path) -> "SpectralImage[RasterioLibImage]":
        image = RasterioLibImage.open(filepath)
        return SpectralImage(image)

    @classmethod
    def from_numpy(cls, array: NDArray[np.floating[Any]]) -> "SpectralImage[MockImage]":
        image = MockImage.open(array)
        return SpectralImage(image)

    @property
    def image(self) -> T:
        return self._image

    @property
    def geometric_shapes(self) -> GeometricShapes:
        return self._geometric_shapes

    @property
    def filepath(self) -> Path:
        return self.image.filepath

    @property
    def metadata(self) -> dict[str, Any]:
        return self.image.metadata

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.image.shape

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def bands(self) -> int:
        return self.image.bands

    @property
    def default_bands(self) -> list[int]:
        return self.image.default_bands

    @property
    def wavelengths(self) -> list[float]:
        return self.image.wavelengths

    @property
    def camera_id(self) -> str:
        return self.image.camera_id

    def to_display(self, equalize: bool = True) -> Image.Image:
        return self.image.to_display(equalize)

    def to_numpy(self, nan_value: float | None = None) -> NDArray[np.floating[Any]]:
        return self.image.to_numpy(nan_value)

    def to_xarray(self) -> "XarrayType":
        return self.image.to_xarray()

    def to_signatures(self, pixels: Pixels | pd.DataFrame | Iterable[CoordinateInput]) -> Signatures:
        pixels = validate_pixel_input(pixels)
        image_arr = self.to_numpy()
        signatures = Signatures.from_array_and_pixels(image_arr, pixels)
        return signatures

    def to_subarray(self, pixels: Pixels | pd.DataFrame | Iterable[CoordinateInput]) -> NDArray[np.floating[Any]]:
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
        image_arr = self.to_numpy()
        return np.nanmean(image_arr, axis=axis)
