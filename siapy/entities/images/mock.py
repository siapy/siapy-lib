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
        if len(array.shape) != 3:
            raise InvalidInputError(
                input_value=array.shape,
                message="Input array must be 3-dimensional (height, width, bands)",
            )

        self._array = array.astype(np.float32)

    @classmethod
    def open(cls, array: NDArray[np.floating[Any]]) -> "MockImage":
        return cls(array=array)

    @property
    def filepath(self) -> Path:
        return Path()

    @property
    def metadata(self) -> dict[str, Any]:
        return {}

    @property
    def shape(self) -> tuple[int, int, int]:
        x = self._array.shape[1]
        y = self._array.shape[0]
        bands = self._array.shape[2]
        return (y, x, bands)

    @property
    def bands(self) -> int:
        return self._array.shape[2]

    @property
    def default_bands(self) -> list[int]:
        if self.bands >= 3:
            return [0, 1, 2]
        return list(range(min(3, self.bands)))

    @property
    def wavelengths(self) -> list[float]:
        return list(range(self.bands))

    @property
    def camera_id(self) -> str:
        return ""

    def to_display(self, equalize: bool = True) -> Image.Image:
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
        if nan_value is not None:
            return np.nan_to_num(self._array, nan=nan_value)
        return self._array.copy()

    def to_xarray(self) -> "XarrayType":
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
