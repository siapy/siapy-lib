import os
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import spectral as sp

from siapy.core import logger
from siapy.core.types import ImageDataType


def save_image(
    image: Annotated[np.ndarray, "The image to save."],
    save_path: Annotated[
        str | Path, "Header file (with '.hdr' extension) name with path."
    ],
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
):
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


def merge_images_by_specter(
    dir_name, image_original, image_to_merge, data_file_name, metadata=None
):
    dfn = parse_data_file_name(data_file_name)
    dir_abs_path = to_absolute_path(os.path.join("outputs", dir_name, dfn.dir_name))
    os.makedirs(dir_abs_path, exist_ok=True)

    if metadata is None:
        metadata = {
            "lines": image_original.rows,
            "samples": image_original.cols,
            "bands": image_original.bands + image_to_merge.bands,
            "data type": image_original.metadata["data type"],
            "default bands": image_original.metadata["default bands"],
            "wavelength": image_original.metadata["wavelength"]
            + image_to_merge.metadata["wavelength"],
            "byte order": image_original.metadata["byte order"],
            "data ignore value": image_original.metadata["data ignore value"],
            "header offset": image_original.metadata["header offset"],
        }

    file_name_hdr = os.path.join(dir_abs_path, dfn.file_name + ".hdr")
    image = sp.envi.create_image(
        file_name_hdr, metadata=metadata, dtype=np.float32, force=True
    )

    image_original_arr = image_original.to_numpy()
    image_to_merge_arr = image_to_merge.to_numpy()

    image_to_merge_arr = rescale_image(image_to_merge_arr, image_original_arr.shape[:2])
    mmap = image.open_memmap(writable=True)
    mmap[:, :, :] = np.concatenate((image_original_arr, image_to_merge_arr), axis=2)

    cfg_image = image_original.config
    cfg_image.name = "merged"
    cfg_image.model = "/"
    return SPImage(image, cfg_image)
