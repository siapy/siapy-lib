import os
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import spectral as sp
from numpy.typing import NDArray

from siapy.core import logger
from siapy.core.exceptions import InvalidInputError
from siapy.core.types import ImageDataType, ImageType
from siapy.entities import SpectralImage
from siapy.entities.helpers import get_signatures_within_convex_hull
from siapy.entities.images import SpectralLibImage
from siapy.transformations.image import rescale
from siapy.utils.validators import validate_image_to_numpy

__all__ = [
    "save_image",
    "create_image",
    "merge_images_by_specter",
    "convert_radiance_image_to_reflectance",
    "calculate_correction_factor_from_panel",
    "blockfy_image",
    "calculate_image_background_percentage",
]


def save_image(
    image: Annotated[NDArray[np.floating[Any]], "The image to save."],
    save_path: Annotated[str | Path, "Header file (with '.hdr' extension) name with path."],
    *,
    metadata: Annotated[
        dict[str, Any] | None,
        "A dict containing ENVI header parameters (e.g., parameters extracted from a source image).",
    ] = None,
    overwrite: Annotated[
        bool,
        "If the associated image file or header already exist and set to True, the files will be overwritten; otherwise, if either of the files exist, an exception will be raised.",
    ] = True,
    dtype: Annotated[
        type[ImageDataType],
        "The numpy data type with which to store the image.",
    ] = np.float32,
) -> None:
    if isinstance(save_path, str):
        save_path = Path(save_path)
    if metadata is None:
        metadata = {}

    os.makedirs(save_path.parent, exist_ok=True)
    sp.envi.save_image(
        hdr_file=save_path,
        image=image,
        dtype=dtype,
        force=overwrite,
        metadata=metadata,
    )
    logger.info(f"Image saved as:  {save_path}")


def create_image(
    image: Annotated[NDArray[np.floating[Any]], "The image to save."],
    save_path: Annotated[str | Path, "Header file (with '.hdr' extension) name with path."],
    *,
    metadata: Annotated[
        dict[str, Any] | None,
        "A dict containing ENVI header parameters (e.g., parameters extracted from a source image).",
    ] = None,
    overwrite: Annotated[
        bool,
        "If the associated image file or header already exist and set to True, the files will be overwritten; otherwise, if either of the files exist, an exception will be raised.",
    ] = True,
    dtype: Annotated[
        type[ImageDataType],
        "The numpy data type with which to store the image.",
    ] = np.float32,
) -> SpectralImage[Any]:
    if isinstance(save_path, str):
        save_path = Path(save_path)
    if metadata is None:
        metadata = {
            "lines": image.shape[0],
            "samples": image.shape[1],
            "bands": image.shape[2],
        }

    os.makedirs(save_path.parent, exist_ok=True)
    spectral_image = sp.envi.create_image(
        hdr_file=save_path,
        metadata=metadata,
        dtype=dtype,
        force=overwrite,
    )
    mmap = spectral_image.open_memmap(writable=True)
    mmap[:, :, :] = image
    logger.info(f"Image created as:  {save_path}")
    return SpectralImage(SpectralLibImage(spectral_image))


def merge_images_by_specter(
    *,
    image_original: Annotated[SpectralImage[Any], "Original image."],
    image_to_merge: Annotated[SpectralImage[Any], "Image which will be merged onto original image."],
    save_path: Annotated[str | Path, "Header file (with '.hdr' extension) name with path."],
    overwrite: Annotated[
        bool,
        "If the associated image file or header already exist and set to True, the files will be overwritten; otherwise, if either of the files exist, an exception will be raised.",
    ] = True,
    dtype: Annotated[
        type[ImageDataType],
        "The numpy data type with which to store the image.",
    ] = np.float32,
    auto_metadata_extraction: Annotated[
        bool,
        "Whether to automatically extract metadata images.",
    ] = True,
) -> SpectralImage[Any]:
    image_original_np = image_original.to_numpy()
    image_to_merge_np = image_to_merge.to_numpy()

    metadata = {
        "lines": image_original.shape[0],
        "samples": image_original.shape[1],
        "bands": image_original.shape[2] + image_to_merge.shape[2],
    }
    if auto_metadata_extraction:
        original_meta = image_original.metadata
        merged_meta = image_to_merge.metadata
        metadata_ext = {}

        metadata_ext["wavelength"] = original_meta.get("wavelength", []) + merged_meta.get("wavelength", [])
        metadata_ext["data type"] = original_meta.get("data type", "")
        metadata_ext["byte order"] = original_meta.get("byte order", "")
        metadata_ext["data ignore value"] = original_meta.get("data ignore value", "")
        metadata_ext["header offset"] = original_meta.get("header offset", 0)
        metadata_ext["interleave"] = original_meta.get("interleave", "")
        metadata_ext["wavelength units"] = original_meta.get("wavelength units", "")
        metadata_ext["acquisition date"] = original_meta.get("acquisition date", "")

        metadata_ext["default bands"] = original_meta.get("default bands", [])
        metadata_ext["default bands additional"] = merged_meta.get("default bands", [])
        metadata_ext["description"] = original_meta.get("description", "")
        # metadata_ext["description additional"] = merged_meta.get("description", "")

        metadata.update(metadata_ext)

    image_to_merge_np = rescale(
        image_to_merge_np,
        (image_original_np.shape[0], image_original_np.shape[1]),
    )
    image_to_merge_np = image_to_merge_np.astype(image_original_np.dtype)
    image_merged = np.concatenate((image_original_np, image_to_merge_np), axis=2)

    return create_image(
        image=image_merged,
        save_path=save_path,
        metadata=metadata,
        overwrite=overwrite,
        dtype=dtype,
    )


def convert_radiance_image_to_reflectance(
    image: SpectralImage[Any],
    panel_correction: NDArray[np.floating[Any]],
    save_path: Annotated[str | Path | None, "Header file (with '.hdr' extension) name with path."] = None,
    **kwargs: Any,
) -> NDArray[np.floating[Any]] | SpectralImage[Any]:
    image_ref_np = image.to_numpy() * panel_correction
    if save_path is None:
        return image_ref_np

    return create_image(
        image=image_ref_np,
        save_path=save_path,
        metadata=image.metadata,
        **kwargs,
    )


def calculate_correction_factor_from_panel(
    image: SpectralImage[Any],
    panel_reference_reflectance: float,
    panel_shape_label: str | None = None,
) -> NDArray[np.floating[Any]]:
    if panel_shape_label:
        panel_shape = image.geometric_shapes.get_by_name(panel_shape_label)
        if not panel_shape:
            raise InvalidInputError(
                input_value={"panel_shape_label": panel_shape_label},
                message="Panel shape label not found.",
            )
        if len(panel_shape) != 1:
            raise InvalidInputError(
                input_value={"panel_shape": panel_shape},
                message="Panel shape label must refer to a single shape.",
            )
        panel_signatures = get_signatures_within_convex_hull(image, panel_shape)[0]
        panel_radiance_mean = panel_signatures.signals.mean()

    else:
        temp_mean = image.mean(axis=(0, 1))
        if not isinstance(temp_mean, np.ndarray):
            raise InvalidInputError(
                input_value={"image": image},
                message=f"Expected image.mean(axis=(0, 1)) to return np.ndarray, but got {type(temp_mean).__name__}.",
            )
        panel_radiance_mean = temp_mean

    panel_reflectance_mean = np.full(image.bands, panel_reference_reflectance)
    panel_correction = panel_reflectance_mean / panel_radiance_mean
    return panel_correction


def blockfy_image(
    image: ImageType,
    p: Annotated[int, "block row size"],
    q: Annotated[int, "block column size"],
) -> list[NDArray[np.floating[Any]]]:
    image_np = validate_image_to_numpy(image)
    # Calculate how many blocks can cover the entire image
    bpr = (image_np.shape[0] - 1) // p + 1  # blocks per row
    bpc = (image_np.shape[1] - 1) // q + 1  # blocks per column

    # Pad array with NaNs so it can be divided by p row-wise and by q column-wise
    image_pad = np.nan * np.ones([p * bpr, q * bpc, image_np.shape[2]])
    image_pad[: image_np.shape[0], : image_np.shape[1], : image_np.shape[2]] = image_np

    image_slices = []
    row_prev = 0

    for row_block in range(bpc):
        row_prev = row_block * p
        column_prev = 0

        for column_block in range(bpr):
            column_prev = column_block * q
            block = image_pad[
                row_prev : row_prev + p,
                column_prev : column_prev + q,
            ]

            if block.shape == (p, q, image_np.shape[2]):
                image_slices.append(block)

    return image_slices


def calculate_image_background_percentage(image: ImageType) -> float:
    image_np = validate_image_to_numpy(image)
    # Check where any of bands include nan values (axis=2) to get positions of background
    mask_nan = np.any(np.isnan(image_np), axis=2)
    # Calculate percentage of background
    percentage = np.sum(mask_nan) / mask_nan.size * 100
    return percentage
