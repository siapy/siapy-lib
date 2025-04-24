import os
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import rioxarray  # noqa  # activate the rio accessor
import spectral as sp
import xarray as xr
from numpy.typing import NDArray

from siapy.core import logger
from siapy.core.exceptions import InvalidInputError
from siapy.core.types import ImageDataType, ImageType
from siapy.entities import SpectralImage
from siapy.entities.images import RasterioLibImage, SpectralLibImage
from siapy.transformations.image import rescale
from siapy.utils.image_validators import validate_image_to_numpy
from siapy.utils.signatures import get_signatures_within_convex_hull

__all__ = [
    "spy_save_image",
    "spy_create_image",
    "spy_merge_images_by_specter",
    "rasterio_save_image",
    "rasterio_create_image",
    "convert_radiance_image_to_reflectance",
    "calculate_correction_factor",
    "calculate_correction_factor_from_panel",
    "blockfy_image",
    "calculate_image_background_percentage",
]


def spy_save_image(
    image: Annotated[ImageType, "The image to save."],
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
    image_np = validate_image_to_numpy(image)
    if isinstance(save_path, str):
        save_path = Path(save_path)
    if metadata is None:
        metadata = {}

    os.makedirs(save_path.parent, exist_ok=True)
    sp.envi.save_image(
        hdr_file=save_path,
        image=image_np,
        dtype=dtype,
        force=overwrite,
        metadata=metadata,
    )
    logger.info(f"Image saved as:  {save_path}")


def spy_create_image(
    image: Annotated[ImageType, "The image to save."],
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
    image_np = validate_image_to_numpy(image)
    if isinstance(save_path, str):
        save_path = Path(save_path)
    if metadata is None:
        metadata = {
            "lines": image_np.shape[0],
            "samples": image_np.shape[1],
            "bands": image_np.shape[2],
        }

    os.makedirs(save_path.parent, exist_ok=True)
    spectral_image = sp.envi.create_image(
        hdr_file=save_path,
        metadata=metadata,
        dtype=dtype,
        force=overwrite,
    )
    mmap = spectral_image.open_memmap(writable=True)
    mmap[:, :, :] = image_np
    logger.info(f"Image created as:  {save_path}")
    return SpectralImage(SpectralLibImage(spectral_image))


def spy_merge_images_by_specter(
    *,
    image_original: Annotated[ImageType, "Original image."],
    image_to_merge: Annotated[ImageType, "Image which will be merged onto original image."],
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
    image_original_np = validate_image_to_numpy(image_original)
    image_to_merge_np = validate_image_to_numpy(image_to_merge)

    metadata = {
        "lines": image_original_np.shape[0],
        "samples": image_original_np.shape[1],
        "bands": image_original_np.shape[2] + image_to_merge_np.shape[2],
    }
    if (
        auto_metadata_extraction
        and isinstance(image_original, SpectralImage)
        and isinstance(image_to_merge, SpectralImage)
    ):
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

    return spy_create_image(
        image=image_merged,
        save_path=save_path,
        metadata=metadata,
        overwrite=overwrite,
        dtype=dtype,
    )


def rasterio_save_image(
    image: ImageType,
    save_path: str | Path,
    *,
    metadata: Annotated[dict[str, Any] | None, "A dict containing additional metadata."] = None,
    overwrite: Annotated[
        bool, "If the file exists and set to True, it will be overwritten; otherwise an exception will be raised."
    ] = True,
    dtype: Annotated[type[ImageDataType], "The numpy data type with which to store the image."] = np.float32,
    **kwargs: Annotated[dict[str, Any], "Additional keyword arguments for rioxarray."],
) -> None:
    """Save an image using rioxarray."""
    image_np = validate_image_to_numpy(image)
    if isinstance(save_path, str):
        save_path = Path(save_path)
    if metadata is None:
        metadata = {}

    os.makedirs(save_path.parent, exist_ok=True)

    if save_path.exists() and not overwrite:
        raise InvalidInputError(
            input_value={"save_path": save_path},
            message=f"File {save_path} already exists and overwrite=False.",
        )

    wavelengths = metadata.get("wavelength", [])
    if not wavelengths:
        wavelengths = np.arange(image_np.shape[2])

    xarray = xr.DataArray(
        data=image_np.transpose(2, 0, 1).astype(dtype),
        dims=["band", "y", "x"],
        coords={
            "y": np.arange(image_np.shape[0]),
            "x": np.arange(image_np.shape[1]),
            "band": wavelengths,
        },
        attrs=metadata,
    )

    xarray.rio.to_raster(save_path, **kwargs)
    logger.info(f"Image saved with rasterio as: {save_path}")


def rasterio_create_image(
    image: Annotated[ImageType, "The image to use."],
    save_path: Annotated[str | Path, "File name with path."],
    *,
    metadata: Annotated[dict[str, Any] | None, "A dict containing additional metadata."] = None,
    overwrite: Annotated[
        bool, "If the file exists and set to True, it will be overwritten; otherwise an exception will be raised."
    ] = True,
    dtype: Annotated[type[ImageDataType], "The numpy data type with which to store the image."] = np.float32,
    **kwargs: Annotated[dict[str, Any], "Additional keyword arguments for rioxarray."],
) -> SpectralImage[Any]:
    """Create and save an image using rioxarray, then return a SpectralImage object."""
    image_np = validate_image_to_numpy(image)
    if isinstance(save_path, str):
        save_path = Path(save_path)

    if metadata is None:
        metadata = {}

    # Save the image first
    rasterio_save_image(
        image=image_np,
        save_path=save_path,
        metadata=metadata,
        overwrite=overwrite,
        dtype=dtype,
        **kwargs,
    )
    logger.info(f"Image created as: {save_path}")
    return SpectralImage(RasterioLibImage.open(save_path))


def convert_radiance_image_to_reflectance(
    image: ImageType,
    panel_correction: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    image_np = validate_image_to_numpy(image)
    return image_np * panel_correction


def calculate_correction_factor(
    panel_radiance_mean: NDArray[np.floating[Any]],
    panel_reference_reflectance: float,
) -> NDArray[np.floating[Any]]:
    if not (0 <= panel_reference_reflectance <= 1):
        raise InvalidInputError(
            input_value={"panel_reference_reflectance": panel_reference_reflectance},
            message="Panel reference reflectance must be between 0 and 1.",
        )

    panel_reflectance_mean = np.full(panel_radiance_mean.shape, panel_reference_reflectance)
    panel_correction = panel_reflectance_mean / panel_radiance_mean
    return panel_correction


def calculate_correction_factor_from_panel(
    image: ImageType,
    panel_reference_reflectance: float,
    panel_shape_label: str | None = None,
) -> NDArray[np.floating[Any]]:
    if panel_shape_label and isinstance(image, SpectralImage):
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
        panel_radiance_mean = panel_signatures.signals.average_signal()

    else:
        image_np = validate_image_to_numpy(image)
        temp_mean = image_np.mean(axis=(0, 1))
        if not isinstance(temp_mean, np.ndarray):
            raise InvalidInputError(
                input_value={"image": image_np},
                message=f"Expected image.mean(axis=(0, 1)) to return np.ndarray, but got {type(temp_mean).__name__}.",
            )
        panel_radiance_mean = temp_mean

    return calculate_correction_factor(
        panel_radiance_mean=panel_radiance_mean,
        panel_reference_reflectance=panel_reference_reflectance,
    )


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
