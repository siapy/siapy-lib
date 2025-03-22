from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Sequence, TypeVar

import numpy as np
from PIL import Image

from ..shapes import GeometricShapes, Shape
from ..signatures import Signatures
from .interfaces import ImageBase
from .spectral_lib import SpectralLibImage

if TYPE_CHECKING:
    from ..pixels import Pixels


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
        return repr(self._image)

    def __str__(self) -> str:
        return str(self._image)

    def __lt__(self, other: "SpectralImage") -> bool:
        return self.filepath.name < other.filepath.name

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SpectralImage):
            return NotImplemented
        return self.filepath.name == other.filepath.name and self._image == other._image

    @classmethod
    def spy_open(
        cls, *, header_path: str | Path, image_path: str | Path | None = None
    ) -> "SpectralImage[SpectralLibImage]":
        image = SpectralLibImage.open(header_path=header_path, image_path=image_path)
        return SpectralImage(image)

    # @classmethod
    # def rasterio_open(cls, filepath: str | Path) -> "SpectralImage":
    #     image = RasterioLib.open(filepath)
    #     return cls(image)

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

    def to_numpy(self, nan_value: float | None = None) -> np.ndarray:
        return self.image.to_numpy(nan_value)

    def to_signatures(self, pixels: "Pixels") -> Signatures:
        image_arr = self.to_numpy()
        signatures = Signatures.from_array_and_pixels(image_arr, pixels)
        return signatures

    def to_subarray(self, pixels: "Pixels") -> np.ndarray:
        image_arr = self.to_numpy()
        u_max = pixels.u().max()
        u_min = pixels.u().min()
        v_max = pixels.v().max()
        v_min = pixels.v().min()
        # create new image
        image_arr_area = np.nan * np.ones((v_max - v_min + 1, u_max - u_min + 1, self.bands))
        # convert original coordinates to coordinates for new image
        v_norm = pixels.v() - v_min
        u_norm = pixels.u() - u_min
        # write values from original image to new image
        image_arr_area[v_norm, u_norm, :] = image_arr[pixels.v(), pixels.u(), :]
        return image_arr_area

    def mean(self, axis: int | tuple[int, ...] | Sequence[int] | None = None) -> float | np.ndarray:
        image_arr = self.to_numpy()
        return np.nanmean(image_arr, axis=axis)
