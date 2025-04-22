from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd
from pydantic import BaseModel, ConfigDict

from siapy.core.exceptions import InvalidInputError
from siapy.core.types import ImageContainerType
from siapy.datasets.schemas import TabularDatasetData
from siapy.entities import Signatures, SpectralImage, SpectralImageSet
from siapy.utils.signatures import get_signatures_within_convex_hull

__all__ = [
    "TabularDataset",
]


class MetaDataEntity(BaseModel):
    image_idx: int
    image_filepath: Path
    camera_id: str
    shape_idx: int
    shape_type: str
    shape_label: str | None
    geometry_idx: int


class TabularDataEntity(MetaDataEntity):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    signatures: Signatures


@dataclass
class TabularDataset:
    def __init__(self, container: ImageContainerType):
        self._image_set = SpectralImageSet([container]) if isinstance(container, SpectralImage) else container
        self._data_entities: list[TabularDataEntity] = []

    def __len__(self) -> int:
        return len(self.data_entities)

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} object with {len(self)} data entities>"

    def __iter__(self) -> Iterator[TabularDataEntity]:
        self._check_data_entities()
        return iter(self.data_entities)

    def __getitem__(self, index: int) -> TabularDataEntity:
        self._check_data_entities()
        return self.data_entities[index]

    @property
    def image_set(self) -> SpectralImageSet:
        return self._image_set

    @property
    def data_entities(self) -> list[TabularDataEntity]:
        return self._data_entities

    def process_image_data(self) -> None:
        self.data_entities.clear()
        for image_idx, image in enumerate(self.image_set):
            for shape_idx, shape in enumerate(image.geometric_shapes.shapes):
                signatures_hull = get_signatures_within_convex_hull(image, shape)
                for geometry_idx, signatures in enumerate(signatures_hull):
                    entity = TabularDataEntity(
                        image_idx=image_idx,
                        shape_idx=shape_idx,
                        geometry_idx=geometry_idx,
                        image_filepath=image.filepath,
                        camera_id=image.camera_id,
                        shape_type=shape.shape_type,
                        shape_label=shape.label,
                        signatures=signatures,
                    )
                    self.data_entities.append(entity)

    def generate_dataset_data(self, mean_signatures: bool = True) -> TabularDatasetData:
        self._check_data_entities()
        signatures_dfs = []
        metadata_dfs = []
        for entity in self.data_entities:
            signatures_df = entity.signatures.to_dataframe().dropna()
            if mean_signatures:
                signatures_df = signatures_df.mean().to_frame().T

            signatures_len = len(signatures_df)
            metadata_df = pd.DataFrame(
                {
                    "image_idx": [str(entity.image_idx)] * signatures_len,
                    "image_filepath": [str(entity.image_filepath)] * signatures_len,
                    "camera_id": [entity.camera_id] * signatures_len,
                    "shape_idx": [str(entity.shape_idx)] * signatures_len,
                    "shape_type": [entity.shape_type] * signatures_len,
                    "shape_label": [entity.shape_label] * signatures_len,
                    "geometry_idx": [str(entity.geometry_idx)] * signatures_len,
                }
            )

            assert list(metadata_df.columns) == list(MetaDataEntity.model_fields.keys()), (
                "Sanity check failed! The columns in metadata_df do not match MetaDataEntity fields."
            )

            signatures_dfs.append(signatures_df)
            metadata_dfs.append(metadata_df)

        signatures_concat = pd.concat(signatures_dfs, ignore_index=True)
        metadata_concat = pd.concat(metadata_dfs, ignore_index=True)
        signatures = Signatures.from_dataframe(signatures_concat)
        return TabularDatasetData(signatures=signatures, metadata=metadata_concat)

    def _check_data_entities(self) -> None:
        if not self.data_entities:
            raise InvalidInputError(
                {
                    "data_entities": self.data_entities,
                    "required_action": f"Run {self.process_image_data.__name__}() to process image set.",
                },
                "No data_entities! You need to process the image set first.",
            )
