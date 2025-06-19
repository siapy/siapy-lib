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
        """Initialize a TabularDataset from spectral image data.

        Creates a tabular dataset that can extract and organize spectral signatures
        from geometric shapes within spectral images for analysis and modeling.

        Args:
            container: Either a single SpectralImage or a SpectralImageSet containing
                multiple spectral images to process.

        Example:
            ```python
            from siapy.entities import SpectralImage
            from siapy.datasets import TabularDataset

            # With a single image
            image = SpectralImage.open_rasterio("path/to/image.tif")
            dataset = TabularDataset(image)

            # With multiple images
            image_set = SpectralImageSet([image1, image2])
            dataset = TabularDataset(image_set)
            ```
        """
        self._image_set = SpectralImageSet([container]) if isinstance(container, SpectralImage) else container
        self._data_entities: list[TabularDataEntity] = []

    def __len__(self) -> int:
        """Get the number of data entities in the dataset.

        Returns:
            The total number of processed data entities (geometric shape instances
            with their associated spectral signatures).

        Note:
            Returns 0 if `process_image_data()` has not been called yet.
        """
        return len(self.data_entities)

    def __str__(self) -> str:
        """Return a string representation of the dataset.

        Returns:
            A formatted string showing the class name and number of data entities.

        Example:
            ```python
            print(dataset)
            # Output: <TabularDataset object with 42 data entities>
            ```
        """
        return f"<{self.__class__.__name__} object with {len(self)} data entities>"

    def __iter__(self) -> Iterator[TabularDataEntity]:
        """Iterate over all data entities in the dataset.

        Returns:
            An iterator yielding TabularDataEntity objects containing spectral
            signatures and associated metadata.

        Raises:
            InvalidInputError: If no data entities exist (image data hasn't been
                processed yet).

        Example:
            ```python
            for entity in dataset:
                print(f"Shape {entity.shape_idx}: {len(entity.signatures)} signatures")
            ```
        """
        self._check_data_entities()
        return iter(self.data_entities)

    def __getitem__(self, index: int) -> TabularDataEntity:
        """Get a data entity by index.

        Args:
            index: Zero-based index of the data entity to retrieve.

        Returns:
            The TabularDataEntity at the specified index.

        Raises:
            InvalidInputError: If no data entities exist (image data hasn't been
                processed yet).
            IndexError: If the index is out of range.

        Example:
            ```python
            first_entity = dataset[0]
            print(f"Camera ID: {first_entity.camera_id}")
            ```
        """
        self._check_data_entities()
        return self.data_entities[index]

    @property
    def image_set(self) -> SpectralImageSet:
        """Get the spectral image set being processed.

        Returns:
            The SpectralImageSet containing all spectral images in this dataset.

        Note:
            This is the original image set provided during initialization,
            possibly converted from a single SpectralImage.
        """
        return self._image_set

    @property
    def data_entities(self) -> list[TabularDataEntity]:
        """Get all processed data entities.

        Returns:
            A list of TabularDataEntity objects, each containing spectral signatures and metadata for a geometric shape instance within the image set.

        Note:
            This list will be empty until `process_image_data()` is called.
            Each entity represents signatures extracted from one geometric shape
            in one image.
        """
        return self._data_entities

    def process_image_data(self) -> None:
        """Extract spectral signatures from geometric shapes in all images.

        Processes each image in the image set, extracting spectral signatures from
        within the convex hull of each geometric shape. Creates TabularDataEntity
        objects containing the signatures along with associated metadata.

        Side Effects:
            - Clears any existing data entities
            - Populates the `data_entities` list with new TabularDataEntity objects
            - Each geometric shape may produce multiple entities if signatures
              are organized into multiple groups

        Note:
            This method must be called before accessing data entities through
            iteration, indexing, or `generate_dataset_data()`.

        Example:
            ```python
            dataset = TabularDataset(image_set)
            dataset.process_image_data()
            print(f"Processed {len(dataset)} data entities")
            ```
        """
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
        """Generate structured dataset data for analysis or export.

        Combines all spectral signatures and metadata from processed data entities
        into a unified TabularDatasetData structure suitable for machine learning
        or statistical analysis.

        Args:
            mean_signatures: If True, compute the mean of all signatures within each
                data entity. If False, include all individual signature measurements.
                Defaults to True.

        Returns:
            A TabularDatasetData object containing: <br>
                - signatures: Combined Signatures object with spectral data <br>
                - metadata: DataFrame with image and shape metadata for each signature <br>
                - (optional) Target values if available in the data entities.

        Raises:
            InvalidInputError: If no data entities exist (image data hasn't been
                processed yet).

        Note:
            The metadata DataFrame columns correspond to MetaDataEntity fields:
            image_idx, image_filepath, camera_id, shape_idx, shape_type,
            shape_label, geometry_idx.

        Example:
            ```python
            dataset.process_image_data()

            # Get averaged signatures per shape
            data = dataset.generate_dataset_data(mean_signatures=True)

            # Get all individual signature measurements
            data_detailed = dataset.generate_dataset_data(mean_signatures=False)

            print(f"Signatures shape: {data.signatures.to_numpy().shape}")
            print(f"Metadata shape: {data.metadata.shape}")
            ```
        """
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
        """Validate that data entities have been processed.

        Raises:
            InvalidInputError: If no data entities exist, indicating that
                `process_image_data()` needs to be called first.

        Note:
            This is an internal validation method used by other methods that
            require processed data entities to function correctly.
        """
        if not self.data_entities:
            raise InvalidInputError(
                {
                    "data_entities": self.data_entities,
                    "required_action": f"Run {self.process_image_data.__name__}() to process image set.",
                },
                "No data_entities! You need to process the image set first.",
            )
