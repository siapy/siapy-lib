from dataclasses import dataclass

from siapy.core.types import ImageContainerType
from siapy.entities import SpectralImage, SpectralImageSet


@dataclass()
class TabularDataset:
    def __init__(self, container: ImageContainerType):
        self._image_set = (
            SpectralImageSet([container])
            if isinstance(container, SpectralImage)
            else container
        )

    @property
    def image_set(self):
        return self._image_set

    def generate(self):
        for image in self.image_set:
            for shape in image.geometric_shapes.shapes:
                image.to_signatures(shape.convex_hull())
