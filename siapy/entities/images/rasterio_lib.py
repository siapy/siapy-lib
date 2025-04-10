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
        self._file = file

    @classmethod
    def open(cls, filepath: str | Path) -> "RasterioLibImage":
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
        return self._file

    @property
    def filepath(self) -> Path:
        return Path(self.file.encoding["source"])

    @property
    def metadata(self) -> dict[str, Any]:
        return self.file.attrs

    @property
    def shape(self) -> tuple[int, int, int]:
        # rioxarray uses (band, y, x) ordering
        return (self.file.y.size, self.file.x.size, self.file.band.size)

    @property
    def rows(self) -> int:
        return self.file.y.size

    @property
    def cols(self) -> int:
        return self.file.x.size

    @property
    def bands(self) -> int:
        return self.file.band.size

    @property
    def default_bands(self) -> list[int]:
        # Most common RGB band combination for satellite imagery
        return list(range(1, min(3, self.bands) + 1))

    @property
    def wavelengths(self) -> list[float]:
        return self.file.band.values

    @property
    def camera_id(self) -> str:
        # Todo: camera_id is not a standard metadata field, should be updated
        return self.metadata.get("camera_id", "")

    def to_display(self, equalize: bool = True) -> Image.Image:
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
        image = np.asarray(self.file.transpose("y", "x", "band").values)
        if nan_value is not None:
            image = np.nan_to_num(image, nan=nan_value)
        return image

    def to_xarray(self) -> "XarrayType":
        return self.file
