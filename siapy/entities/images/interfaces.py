from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

__all__ = [
    "ImageBase",
]


class ImageBase(ABC):
    @classmethod
    @abstractmethod
    def open(cls: type["ImageBase"], *args: Any, **kwargs: Any) -> "ImageBase":
        pass

    @property
    @abstractmethod
    def filepath(self) -> Path:
        pass

    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        pass

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int, int]:
        pass

    @property
    @abstractmethod
    def bands(self) -> int:
        pass

    @property
    @abstractmethod
    def default_bands(self) -> list[int]:
        pass

    @property
    @abstractmethod
    def wavelengths(self) -> list[float]:
        pass

    @property
    @abstractmethod
    def camera_id(self) -> str:
        pass

    @abstractmethod
    def to_display(self, equalize: bool = True) -> Image.Image:
        pass

    @abstractmethod
    def to_numpy(self, nan_value: float | None = None) -> np.ndarray:
        pass
