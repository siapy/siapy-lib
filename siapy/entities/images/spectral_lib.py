# mypy: ignore-errors
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import spectral as sp
import xarray as xr
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
        self._file = file

    @classmethod
    def open(cls, *, header_path: str | Path, image_path: str | Path | None = None) -> "SpectralLibImage":
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
        return self._file

    @property
    def filepath(self) -> Path:
        return Path(self.file.filename)

    @property
    def metadata(self) -> dict[str, Any]:
        return self.file.metadata

    @property
    def shape(self) -> tuple[int, int, int]:
        rows = self.file.nrows
        samples = self.file.ncols
        bands = self.file.nbands
        return (rows, samples, bands)

    @property
    def rows(self) -> int:
        return self.file.nrows

    @property
    def cols(self) -> int:
        return self.file.ncols

    @property
    def bands(self) -> int:
        return self.file.nbands

    @property
    def default_bands(self) -> list[int]:
        db = self.metadata.get("default bands", [])
        return list(map(int, db))

    @property
    def wavelengths(self) -> list[float]:
        wavelength_data = self.metadata.get("wavelength", [])
        return list(map(float, wavelength_data))

    @property
    def description(self) -> dict[str, Any]:
        description_str = self.metadata.get("description", {})
        return _parse_description(description_str)

    @property
    def camera_id(self) -> str:
        return self.description.get("ID", "")

    def to_display(self, equalize: bool = True) -> Image.Image:
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

    def to_numpy(self, nan_value: float | None = None) -> np.ndarray:
        image = self.file[:, :, :]
        if nan_value is not None:
            image = self._remove_nan(image, nan_value)
        return image

    def _remove_nan(self, image: np.ndarray, nan_value: float = 0.0) -> np.ndarray:
        # TODO: Remove this function and substitute with np.nan_to_num
        image_mask = np.bitwise_not(np.bool_(np.isnan(image).sum(axis=2)))
        image[~image_mask] = nan_value
        return image

    def to_xarray(self) -> "XarrayType":
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
