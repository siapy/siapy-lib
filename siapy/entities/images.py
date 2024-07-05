# mypy: ignore-errors
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator

import numpy as np
import spectral as sp
from PIL import Image, ImageOps

from .shapes import Shape
from .signatures import Signatures

if TYPE_CHECKING:
    from ..core.types import SpectralType
    from .pixels import Pixels


@dataclass
class GeometricShapes:
    def __init__(
        self, image: "SpectralImage", geometric_shapes: list["Shape"] | None = None
    ):
        self._image = image
        self._geometric_shapes = (
            geometric_shapes if geometric_shapes is not None else []
        )

    def __iter__(self) -> Iterator[Shape]:
        return iter(self.shapes)

    def __getitem__(self, index: int) -> Shape:
        return self.shapes[index]

    def __setitem__(self, index: int, shape: Shape):
        self._check_shape_type(shape)
        self._geometric_shapes[index] = shape

    def __len__(self) -> int:
        return len(self._geometric_shapes)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, GeometricShapes):
            return NotImplemented
        return (
            self._geometric_shapes == other._geometric_shapes
            and self._image == other._image
        )

    @property
    def shapes(self) -> list["Shape"]:
        return self._geometric_shapes.copy()

    @shapes.setter
    def shapes(self, shapes: list["Shape"]):
        self._check_shape_type(shapes)
        self._geometric_shapes = shapes

    def append(self, shape: "Shape"):
        self._check_shape_type(shape)
        self._geometric_shapes.append(shape)

    def extend(self, shapes: Iterable["Shape"]):
        self._check_shape_type(shapes)
        self._geometric_shapes.extend(shapes)

    def insert(self, index: int, shape: "Shape"):
        self._check_shape_type(shape)
        self._geometric_shapes.insert(index, shape)

    def remove(self, shape: "Shape"):
        self._check_shape_type(shape)
        self._geometric_shapes.remove(shape)

    def pop(self, index: int = -1) -> "Shape":
        return self._geometric_shapes.pop(index)

    def clear(self):
        self._geometric_shapes.clear()

    def index(self, shape: "Shape", start: int = 0, stop: int = sys.maxsize) -> int:
        self._check_shape_type(shape)
        return self._geometric_shapes.index(shape, start, stop)

    def count(self, shape: "Shape") -> int:
        self._check_shape_type(shape)
        return self._geometric_shapes.count(shape)

    def reverse(self):
        self._geometric_shapes.reverse()

    def sort(self, key: Any = None, reverse: bool = False):
        self._geometric_shapes.sort(key=key, reverse=reverse)

    def get_by_name(self, name: str) -> Shape | None:
        names = [shape.label for shape in self.shapes]
        if name in names:
            index = names.index(name)
            return self.shapes[index]

    def _check_shape_type(self, shapes: Shape | Iterable[Shape]):
        if isinstance(shapes, Shape):
            return
        if not isinstance(shapes, Iterable):
            raise ValueError(
                "Shapes must be an instance of Shape or an iterable of Shape instances."
            )
        if not all(isinstance(shape, Shape) for shape in shapes):
            raise ValueError("All items must be instances of Shape subclass.")


@dataclass
class SpectralImage:
    def __init__(
        self, sp_file: "SpectralType", geometric_shapes: list["Shape"] | None = None
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
        return (
            self.filepath.name == other.filepath.name
            and self._sp_file == other._sp_file
        )

    @classmethod
    def envi_open(
        cls, *, header_path: str | Path, image_path: str | Path | None = None
    ) -> "SpectralImage":
        sp_file = sp.envi.open(file=header_path, image=image_path)
        if isinstance(sp_file, sp.io.envi.SpectralLibrary):
            raise ValueError("Opened file of type SpectralLibrary")
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

    def mean(self, axis: int | tuple[int] | None = None) -> float | np.ndarray:
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
                value = [
                    float(v) if v.replace(".", "", 1).isdigit() else v
                    for v in value.split(",")
                ]
            elif value.isdigit():
                value = int(value)
            elif value.replace(".", "", 1).isdigit():
                value = float(value)
            data_dict[key] = value
        return data_dict

    try:
        return _parse()
    except ValueError as e:
        raise ValueError(f"Error parsing description: {e}") from e
    except KeyError as e:
        raise KeyError(f"Missing key in description: {e}") from e
    except Exception as e:
        raise Exception(f"Unexpected error parsing description: {e}") from e
