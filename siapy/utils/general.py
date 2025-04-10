import inspect
import multiprocessing
import random
import re
import types
from functools import partial
from pathlib import Path
from typing import Any, Callable, Generator, Iterable, Optional

import numpy as np

from siapy.core import logger
from siapy.core.exceptions import InvalidInputError

__all__ = [
    "initialize_object",
    "initialize_function",
    "ensure_dir",
    "get_number_cpus",
    "dict_zip",
    "get_increasing_seq_indices",
    "set_random_seed",
    "get_classmethods",
    "match_iterable_items_by_regex",
]


def initialize_object(
    module: types.ModuleType | Any,
    module_name: str,
    module_args: Optional[dict[str, Any]] = None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    module_args = module_args or {}
    assert not set(kwargs).intersection(module_args), "Overwriting kwargs given in config file is not allowed"
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
    assert not set(kwargs).intersection(module_args), "Overwriting kwargs given in config file is not allowed"
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
        raise InvalidInputError(input_value=parallelize, message="Define accurate number of CPUs.")
    return parallelize


def dict_zip(
    *dicts: dict[str, Any],
) -> Generator[tuple[str, Any, Any], None, None]:
    if not dicts:
        return

    n = len(dicts[0])
    if any(len(d) != n for d in dicts):
        raise InvalidInputError(input_value=dicts, message="Arguments must have the same length.")

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


def set_random_seed(seed: int | None) -> None:
    random.seed(seed)
    np.random.seed(seed)


def get_classmethods(class_obj: Any) -> list[str]:
    return [
        member[0]
        for member in inspect.getmembers(class_obj, predicate=inspect.ismethod)
        if member[1].__self__ == class_obj
    ]


def match_iterable_items_by_regex(
    iterable1: Iterable[str], iterable2: Iterable[str], regex: str = r""
) -> tuple[list[tuple[str, str]], list[tuple[int, int]]]:
    pattern = re.compile(regex)
    matches = []
    indices = []
    for idx1, item1 in enumerate(iterable1):
        match1 = pattern.search(item1)
        logger.debug("match1: %s", match1)
        if match1:
            substring1 = match1.group()
            for idx2, item2 in enumerate(iterable2):
                match2 = pattern.search(item2)
                logger.debug("match2: %s", match2)
                if match2 and substring1 == match2.group():
                    matches.append((item1, item2))
                    indices.append((idx1, idx2))
                    logger.info("Matched items: %s -> %s", item1, item2)
    return matches, indices
