from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from pydantic import BaseModel, ConfigDict

from siapy.core.types import ImageContainerType
from siapy.entities import Pixels, Signatures, SpectralImage, SpectralImageSet
from siapy.models.schemas import ClassificationTarget, RegressionTarget


class MetaDataEntity(BaseModel):
    image_idx: int
    image_filepath: Path
    camera_id: str
    shape_type: str
    shape_label: str | None


class TabularDataEntity(MetaDataEntity):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    signatures: Signatures


class DatasetDataFrame(pd.DataFrame):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    @property
    def _constructor(self) -> type["DatasetDataFrame"]:
        # Ensure compatibility with pandas methods by overriding _constructor
        return DatasetDataFrame

    def signals(self) -> "DatasetDataFrame":
        excluded_cols = [Pixels.coords.U, Pixels.coords.V] + list(
            MetaDataEntity.model_fields.keys()
        )
        signal_cols = [col for col in self.columns if col not in excluded_cols]
        return self._return_dataframe(signal_cols)

    def metadata(self) -> "DatasetDataFrame":
        metadata_cols = list(MetaDataEntity.model_fields.keys())
        return self._return_dataframe(metadata_cols)

    def pixels(self) -> "DatasetDataFrame":
        pixels_cols = [Pixels.coords.U, Pixels.coords.V]
        return self._return_dataframe(pixels_cols)

    def _return_dataframe(self, columns: list[str]) -> "DatasetDataFrame":
        return DatasetDataFrame(self[columns].copy())

    def generate_classification_target(
        self, column_names: str | list[str]
    ) -> ClassificationTarget:
        if isinstance(column_names, str):
            column_names = [column_names]
        # create one column labels from multiple columns
        label = self[column_names].apply(tuple, axis=1)
        # encode to numbers
        encoded_np, encoding_np = pd.factorize(label)
        encoded = pd.Series(encoded_np, name="encoded")
        encoding = pd.Series(encoding_np, name="encoding")
        return ClassificationTarget(label=label, value=encoded, encoding=encoding)

    def generate_regression_target(self, column_name: str) -> RegressionTarget:
        return RegressionTarget(name=column_name, value=self[column_name])


@dataclass()
class TabularDataset:
    def __init__(self, container: ImageContainerType):
        self._image_set = (
            SpectralImageSet([container])
            if isinstance(container, SpectralImage)
            else container
        )
        self._data_entities: list[TabularDataEntity] = []

    def __len__(self) -> int:
        return len(self.data_entities)

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} object with {len(self)} data entities>"

    def __iter__(self) -> Iterator[TabularDataEntity]:
        self._check_data_entities()
        return iter(self.data_entities)

    def __getitem__(self, index) -> TabularDataEntity:
        self._check_data_entities()
        return self.data_entities[index]

    @property
    def image_set(self) -> SpectralImageSet:
        return self._image_set

    @property
    def data_entities(self) -> list[TabularDataEntity]:
        return self._data_entities

    def process_image_data(self):
        for idx, image in enumerate(self.image_set):
            for shape in image.geometric_shapes.shapes:
                signatures = image.to_signatures(shape.convex_hull())
                entity = TabularDataEntity(
                    image_idx=idx,
                    image_filepath=image.filepath,
                    camera_id=image.camera_id,
                    shape_type=shape.shape_type,
                    shape_label=shape.label,
                    signatures=signatures,
                )
                self.data_entities.append(entity)

    def generate_dataset(self, mean_signatures=True) -> DatasetDataFrame:
        self._check_data_entities()
        entity_dfs = []
        for entity in self.data_entities:
            signatures_df = (
                pd.DataFrame([entity.signatures.df().dropna().mean()])
                if mean_signatures
                else entity.signatures.df()
            )
            signatures_len = len(signatures_df)
            repeated_info_df = pd.DataFrame(
                {
                    "image_idx": [str(entity.image_idx)] * signatures_len,
                    "image_filepath": [str(entity.image_filepath)] * signatures_len,
                    "camera_id": [entity.camera_id] * signatures_len,
                    "shape_type": [entity.shape_type] * signatures_len,
                    "shape_label": [entity.shape_label] * signatures_len,
                }
            )

            assert (
                list(repeated_info_df.columns)
                == list(MetaDataEntity.model_fields.keys())
            ), "Sanity check failed! The columns in repeated_info_df do not match metadata fields."

            full_entity_df = pd.concat(
                [repeated_info_df, signatures_df.reset_index(drop=True)],
                axis=1,
            )
            entity_dfs.append(full_entity_df)
        return DatasetDataFrame(pd.concat(entity_dfs, ignore_index=True))

    def _check_data_entities(self):
        if not self.data_entities:
            raise ValueError(
                f"No data_entities! You need to process image set first."
                f"by running {self.process_image_data.__name__}() function."
            )
