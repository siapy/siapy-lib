# mypy: ignore-errors
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import spectral as sp
from PIL import Image, ImageOps

from siapy.core.exceptions import InvalidFilepathError, InvalidInputError

from .shapes import GeometricShapes, Shape
from .signatures import Signatures

if TYPE_CHECKING:
    from ..core.types import SpectralType
    from .pixels import Pixels


__all__ = [
    "SpectralImage",
]


@dataclass
class SpectralImage:
    def __init__(
        self,
        sp_file: "SpectralType",
        geometric_shapes: list["Shape"] | None = None,
    ):
        self._sp_file = sp_file
        self._geometric_shapes = GeometricShapes(self, geometric_shapes)

    def __repr__(self) -> str:
        return repr(self._sp_file)

    def __str__(self) -> str:
        return str(self._sp_file)

    def __lt__(self, other: "SpectralImage") -> bool:
        return self.filepath.name < other.filepath.name

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SpectralImage):
            return NotImplemented
        return self.filepath.name == other.filepath.name and self._sp_file == other._sp_file

    @classmethod
    def envi_open(cls, *, header_path: str | Path, image_path: str | Path | None = None) -> "SpectralImage":
        if not Path(header_path).exists():
            raise InvalidFilepathError(str(header_path))
        sp_file = sp.envi.open(file=header_path, image=image_path)
        if isinstance(sp_file, sp.io.envi.SpectralLibrary):
            raise InvalidInputError(
                {
                    "file_type": type(sp_file).__name__,
                },
                "Opened file of type SpectralLibrary",
            )
        return cls(sp_file)

    @property
    def file(self) -> "SpectralType":
        return self._sp_file

    @property
    def filepath(self) -> Path:
        return Path(self._sp_file.filename)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._sp_file.metadata

    @property
    def shape(self) -> tuple[int, int, int]:
        rows = self._sp_file.nrows
        samples = self._sp_file.ncols
        bands = self._sp_file.nbands
        return (rows, samples, bands)

    @property
    def rows(self) -> int:
        return self._sp_file.nrows

    @property
    def cols(self) -> int:
        return self._sp_file.ncols

    @property
    def bands(self) -> int:
        return self._sp_file.nbands

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

    @property
    def geometric_shapes(self) -> GeometricShapes:
        return self._geometric_shapes

    def to_display(self, equalize: bool = True) -> Image.Image:
        max_uint8 = 255.0
        image_3ch = self._sp_file.read_bands(self.default_bands)
        image_3ch = self._remove_nan(image_3ch, nan_value=0)
        image_3ch[:, :, 0] = image_3ch[:, :, 0] / image_3ch[:, :, 0].max() / max_uint8
        image_3ch[:, :, 1] = image_3ch[:, :, 1] / (image_3ch[:, :, 1].max() / max_uint8)
        image_3ch[:, :, 2] = image_3ch[:, :, 2] / (image_3ch[:, :, 2].max() / max_uint8)
        image = Image.fromarray(image_3ch.astype("uint8"))
        if equalize:
            image = ImageOps.equalize(image)
        return image

    def to_numpy(self, nan_value: float | None = None) -> np.ndarray:
        image = self._sp_file[:, :, :]
        if nan_value is not None:
            image = self._remove_nan(image, nan_value)
        return image

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

    def _remove_nan(self, image: np.ndarray, nan_value: float = 0.0) -> np.ndarray:
        image_mask = np.bitwise_not(np.bool_(np.isnan(image).sum(axis=2)))
        image[~image_mask] = nan_value
        return image


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
