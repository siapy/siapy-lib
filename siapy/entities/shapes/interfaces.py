from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from siapy.entities import Pixels


@dataclass
class ShapeBase(ABC):
    def __init__(self, pixels: "Pixels", label: str | None = None):
        self._pixels = pixels
        self._label = label if label is not None else ""

    @property
    def pixels(self) -> "Pixels":
        return self._pixels

    @property
    def label(self) -> str:
        return self._label

    @property
    def shape_type(self) -> str:
        return self.__class__.__name__.lower()

    @abstractmethod
    def convex_hull(self) -> "Pixels":
        pass
