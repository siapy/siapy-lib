from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import rasterio
from PIL import Image, ImageOps
from rasterio.enums import Resampling

from siapy.core.exceptions import InvalidFilepathError, InvalidInputError

from .interfaces import ImageBase

__all__ = [
    "RasterioLibImage",
]


@dataclass
class RasterioLibImage(ImageBase):
    def __init__(self, file: rasterio.DatasetReader):
        self._file = file

    @classmethod
    def open(cls, filepath: str | Path) -> "RasterioLibImage":
        if not Path(filepath).exists():
            raise InvalidFilepathError(filepath)
        try:
            raster_file = rasterio.open(filepath)
            return cls(raster_file)
        except Exception as e:
            raise InvalidInputError(
                {"filepath": filepath},
                f"Failed to open raster file: {e}",
            ) from e

    @property
    def file(self) -> rasterio.DatasetReader:
        return self._file

    @property
    def filepath(self) -> Path:
        return Path(self.file.name)

    @property
    def metadata(self) -> dict[str, Any]:
        return self.file.tags()

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.file.height, self.file.width, self.file.count)

    @property
    def bands(self) -> int:
        return self.file.count

    @property
    def default_bands(self) -> list[int]:
        # Most common RGB band combination for satellite imagery
        if self.bands >= 3:
            return [0, 1, 2]
        return list(range(min(3, self.bands)))

    @property
    def wavelengths(self) -> list[float]:
        # Try to get wavelengths from metadata
        wavelengths = []
        for i in range(1, self.bands + 1):
            band = self.file.tags(i).get("wavelength")
            if band:
                wavelengths.append(float(band))
            else:
                wavelengths.append(float(i))
        return wavelengths

    @property
    def camera_id(self) -> str:
        return self.metadata.get("camera_id", "")

    def to_display(self, equalize: bool = True) -> Image.Image:
        # Read default bands and scale to 8-bit
        display_bands = [self.file.read(i + 1) for i in self.default_bands]
        image_3ch = np.dstack(display_bands)

        # Normalize and scale to 0-255
        image_3ch = ((image_3ch - image_3ch.min()) * (255.0 / (image_3ch.max() - image_3ch.min()))).astype(np.uint8)

        image = Image.fromarray(image_3ch)
        if equalize:
            image = ImageOps.equalize(image)
        return image

    def to_numpy(self, nan_value: float | None = None) -> np.ndarray:
        # Read all bands
        image = np.dstack([self.file.read(i) for i in range(1, self.bands + 1)])

        if nan_value is not None:
            image[np.isnan(image)] = nan_value

        return image
