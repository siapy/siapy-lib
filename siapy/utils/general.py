import multiprocessing
import types
from functools import partial
from pathlib import Path
from typing import Any, Callable, Generator, Optional

import numpy as np
from PIL.Image import Image

from siapy.core.types import ImageType
from siapy.entities import SpectralImage


def initialize_object(
    module: types.ModuleType | Any,
    module_name: str,
    module_args: Optional[dict[str, Any]] = None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    module_args = module_args or {}
    assert not set(kwargs).intersection(
        module_args
    ), "Overwriting kwargs given in config file is not allowed"
    module_args.update(kwargs)
    return getattr(module, module_name)(*args, **module_args)


def initialize_function(
    module: types.ModuleType | Any,
    module_name: str,
    module_args: Optional[dict[str, Any]] = None,
    *args: Any,
    **kwargs: Any,
) -> Callable[..., Any]:
    module_args = module_args or {}
    assert not set(kwargs).intersection(
        module_args
    ), "Overwriting kwargs given in config file is not allowed"
    module_args.update(kwargs)
    return partial(getattr(module, module_name), *args, **module_args)


def ensure_dir(dirname: str | Path) -> Path:
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)
    return dirname


def get_number_cpus(parallelize: int = -1) -> int:
    num_cpus: int = multiprocessing.cpu_count()
    if parallelize == -1:
        parallelize = num_cpus
    elif 1 <= parallelize <= num_cpus:
        pass
    elif parallelize > num_cpus:
        parallelize = num_cpus
    else:
        raise ValueError("Define accurate number of cpus")
    return parallelize


def dict_zip(*dicts: dict[str, Any]) -> Generator[tuple[str, Any, Any], None, None]:
    if not dicts:
        return

    n = len(dicts[0])
    if any(len(d) != n for d in dicts):
        raise ValueError("arguments must have the same length")

    for key, first_val in dicts[0].items():
        yield key, first_val, *(other[key] for other in dicts[1:])


def get_increasing_seq_indices(values_list: list[int]) -> list[int]:
    indices = []
    last_value = 0
    for idx, value in enumerate(values_list):
        if value > last_value:
            last_value = value
            indices.append(idx)
    return indices


def validate_and_convert_image(image: ImageType) -> np.ndarray:
    if isinstance(image, SpectralImage):
        image_display = np.array(image.to_display())
    elif isinstance(image, Image):
        image_display = np.array(image)
    elif (
        isinstance(image, np.ndarray) and len(image.shape) == 3 and image.shape[-1] == 3
    ):
        image_display = image.copy()
    else:
        raise ValueError("Image must be convertable to 3d numpy array.")
    return image_display
