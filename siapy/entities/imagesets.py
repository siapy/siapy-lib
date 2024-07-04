from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from rich.progress import track

from siapy.core import logger

from .images import SpectralImage


@dataclass
class SpectralImageSet:
    def __init__(self, spectral_images: list[SpectralImage] | None = None):
        self._images = spectral_images if spectral_images is not None else []

    def __len__(self) -> int:
        return len(self.images)

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} object with {len(self)} spectral images>"

    def __iter__(self) -> Iterator[SpectralImage]:
        return iter(self.images)

    def __getitem__(self, index) -> SpectralImage:
        return self.images[index]

    @classmethod
    def from_paths(
        cls,
        *,
        header_paths: list[str | Path],
        image_paths: list[str | Path] | None = None,
    ):
        if image_paths is not None and len(header_paths) != len(image_paths):
            raise ValueError("The length of hdr_paths and img_path must be equal.")

        if image_paths is None:
            spectral_images = [
                SpectralImage.envi_open(header_path=hdr_path)
                for hdr_path in track(
                    header_paths, description="Loading spectral images..."
                )
            ]
        else:
            spectral_images = [
                SpectralImage.envi_open(header_path=hdr_path, image_path=img_path)
                for hdr_path, img_path in track(
                    zip(header_paths, image_paths),
                    description="Loading spectral images...",
                )
            ]
        logger.info("Spectral images loaded into memory.")
        return cls(spectral_images)

    @property
    def images(self) -> list[SpectralImage]:
        return self._images
