from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import rioxarray
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
            raise InvalidInputError({"file_type": type(raster).__name__}, "Expected DataArray, got Dataset")

        return cls(raster)

    @property
    def file(self) -> "XarrayType":
        return self._file

    @property
    def filepath(self) -> Path:
        return Path(self.file.encoding["source"])

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self.file.attrs)

    @property
    def shape(self) -> tuple[int, int, int]:
        # rioxarray uses (band, y, x) ordering
        return (self.file.y.size, self.file.x.size, self.file.band.size)

    @property
    def bands(self) -> int:
        return self.file.band.size

    @property
    def default_bands(self) -> list[int]:
        # Most common RGB band combination for satellite imagery
        if self.bands >= 3:
            return [0, 1, 2]
        return list(range(min(3, self.bands)))

    @property
    def wavelengths(self) -> list[float]:
        # Try to get wavelengths from band attributes
        wavelengths = []
        for band_idx in range(self.bands):
            band_data = self.file.sel(band=band_idx + 1)
            wave = band_data.attrs.get("wavelength")
            if wave:
                wavelengths.append(float(wave))
            else:
                wavelengths.append(float(band_idx + 1))
        return wavelengths

    @property
    def camera_id(self) -> str:
        return self.metadata.get("camera_id", "")

    def to_display(self, equalize: bool = True) -> Image.Image:
        selected_bands = [i + 1 for i in self.default_bands]  # Adjust for 1-indexed bands
        bands_data = self.file.sel(band=selected_bands)
        image_3ch = bands_data.transpose("y", "x", "band").values
        image_3ch_clean = np.nan_to_num(np.asarray(image_3ch))
        min_val = np.nanmin(image_3ch_clean)
        max_val = np.nanmax(image_3ch_clean)

        image_scaled = ((image_3ch_clean - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8)

        image = Image.fromarray(image_scaled)
        if equalize:
            image = ImageOps.equalize(image)
        return image

    def to_numpy(self, nan_value: float | None = None) -> np.ndarray:
        image = np.moveaxis(np.asarray(self.file.values), 0, -1)
        if nan_value is not None:
            image = np.nan_to_num(image, nan=nan_value)
        return image
