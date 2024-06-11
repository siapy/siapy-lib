from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel
from rich.progress import track

from siapy.core import logger
from .images import SpectralImage


class SpectralImageContainerConfig(BaseModel):
    img_paths: list[str | Path]
    hdr_paths: list[str | Path]


@dataclass
class SpectralImageContainer:
    def __init__(self, config: SpectralImageContainerConfig):
        self._config = config.model_copy()
        self._images: list[SpectralImage] = []

    def __len__(self) -> int:
        return len(self.images)

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} object with {len(self)} spectral images>"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config!r})"

    def __iter__(self) -> Iterator[SpectralImage]:
        return iter(self.images)

    def __getitem__(self, index) -> SpectralImage:
        return self.images[index]

    @property
    def config(self) -> SpectralImageContainerConfig:
        return self._config

    @property
    def images(self):
        return self._images

    def _import_spectral_images(self):
        self._images = [
            SpectralImage.envi_open(hdr_path, img_path)
            for hdr_path, img_path in track(
                zip(self.config.hdr_paths, self.config.img_paths),
                "[green]Loading spectral images...",
            )
        ]

    def _sanity_check(self):
        if len(self.config.img_paths) != len(self.config.hdr_paths):
            raise ValueError("Image paths must be the same length.")

    def load(self) -> "SpectralImageContainer":
        self._sanity_check()
        self._import_spectral_images()
        logger.info("Spectral images loaded into memory.")
        return self


@dataclass
class SpectralImageMultiCameraContainer:
    def __init__(self, *containers: SpectralImageContainer):
        self.containers = list(containers)

    def __len__(self) -> int:
        return len(self.containers)

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} object with {len(self)} {SpectralImageContainer.__name__}>"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(containers={self.containers!r})"

    def __iter__(self) -> Iterator[tuple[SpectralImage, ...]]:
        iterators = [iter(container) for container in self.containers]
        for images in zip_longest(*iterators):
            yield images
