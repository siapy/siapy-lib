import os
import warnings

import numpy as np
import pandas as pd
import spectral as sp
from skimage import transform

from siapy.entities import SPImage
from siapy.utils.utils import get_logger, parse_data_file_name, to_absolute_path

logger = get_logger(name="image_utils")


def average_signatures(area_of_signatures):
    if area_of_signatures is not None:
        x_center = (
            area_of_signatures.x.min()
            + (area_of_signatures.x.max() - area_of_signatures.x.min()) / 2
        )
        y_center = (
            area_of_signatures.y.min()
            + (area_of_signatures.y.max() - area_of_signatures.y.min()) / 2
        )
        signatures_mean = [list(area_of_signatures.signature.mean())]

        data = {"x": int(x_center), "y": int(y_center), "signature": signatures_mean}
        return pd.DataFrame(data, columns=["x", "y", "signature"])
    else:
        return list()


def limit_to_bounds(image_shape):
    y_max = image_shape[0]
    x_max = image_shape[1]

    def _limit(points):
        points = points[
            (points.x >= 0) & (points.y >= 0) & (points.x < x_max) & (points.y < y_max)
        ]
        return points

    return _limit


def save_image(config, image, data_file_name, metadata=None):
    dfn = parse_data_file_name(data_file_name)
    dir_abs_path = to_absolute_path(os.path.join("outputs", config.name, dfn.dir_name))
    os.makedirs(dir_abs_path, exist_ok=True)
    file_name_hdr = os.path.join(dir_abs_path, dfn.file_name + ".hdr")
    sp.envi.save_image(
        file_name_hdr, image, dtype=np.float32, metadata=metadata, force=True
    )
    logger.info(f"Images saved as:  {file_name_hdr}")


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


def rescale_image(image, output_size):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    h, w = image.shape[:2]
    if isinstance(output_size, int):
        if h > w:
            new_h, new_w = output_size * h / w, output_size
        else:
            new_h, new_w = output_size, output_size * w / h
    else:
        new_h, new_w = output_size

    new_h, new_w = int(new_h), int(new_w)

    # to suppresss the warning: "All-Nan slice encountered"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"All-NaN slice encountered")
        resized_image = transform.resize(image, (new_h, new_w))

    return resized_image
