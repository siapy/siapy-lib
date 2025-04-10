from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
from rich.progress import track

from siapy.core import logger
from siapy.core.exceptions import InvalidInputError
from siapy.entities import SpectralImage

__all__ = [
    "SpectralImageSet",
]


@dataclass
class SpectralImageSet:
    def __init__(self, spectral_images: list[SpectralImage[Any]] | None = None):
        self._images = spectral_images if spectral_images is not None else []

    def __len__(self) -> int:
        return len(self.images)

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} object with {len(self)} spectral images>"

    def __iter__(self) -> Iterator[SpectralImage[Any]]:
        return iter(self.images)

    def __getitem__(self, index: int) -> SpectralImage[Any]:
        return self.images[index]

    @classmethod
    def spy_open(
        cls,
        *,
        header_paths: Sequence[str | Path],
        image_paths: Sequence[str | Path] | None = None,
    ) -> "SpectralImageSet":
        if image_paths is not None and len(header_paths) != len(image_paths):
            raise InvalidInputError(
                {
                    "header_paths_length": len(header_paths),
                    "image_paths_length": len(image_paths),
                },
                "The length of hdr_paths and img_path must be equal.",
            )

        if image_paths is None:
            spectral_images = [
                SpectralImage.spy_open(header_path=hdr_path)
                for hdr_path in track(header_paths, description="Loading spectral images...")
            ]
        else:
            spectral_images = [
                SpectralImage.spy_open(header_path=hdr_path, image_path=img_path)
                for hdr_path, img_path in track(
                    zip(header_paths, image_paths),
                    description="Loading spectral images...",
                )
            ]
        logger.info("Spectral images loaded into memory.")
        return cls(spectral_images)

    @classmethod
    def rasterio_open(
        cls,
        *,
        filepaths: Sequence[str | Path],
    ) -> "SpectralImageSet":
        spectral_images = [
            SpectralImage.rasterio_open(filepath)
            for filepath in track(filepaths, description="Loading raster images...")
        ]
        logger.info("Raster images loaded into memory.")
        return cls(spectral_images)

    @property
    def images(self) -> list[SpectralImage[Any]]:
        return self._images

    @property
    def cameras_id(self) -> list[str]:
        return list({image.camera_id for image in self.images})

    def images_by_camera_id(self, camera_id: str) -> list[SpectralImage[Any]]:
        ids = np.array([image.camera_id for image in self.images])
        indices = np.nonzero(ids == camera_id)[0]
        return [image for idx, image in enumerate(self.images) if idx in indices]

    def sort(self, key: Any = None, reverse: bool = False) -> None:
        self.images.sort(key=key, reverse=reverse)
