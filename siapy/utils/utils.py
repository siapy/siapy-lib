import logging
import multiprocessing
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from timeit import default_timer

from siapy.core import logger


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent.resolve()


def to_absolute_path(path):
    return os.path.join(get_project_root(), path)


# def save_data(config, data, data_file_name, saver="pickle"):
#     # TODO: change first subfolder to name of a project used
#     # remove file extensions from those functions
#     # create different pattern for load and save of data
#     """Save data

#     Args:
#         config (DictConfig): hydra config file
#         data: data to be stored
#         data_dir_name (str): name of directory where data will be stored
#     """
#     dfn = parse_data_file_name(data_file_name)
#     dir_abs_path = os.path.join(
#         get_project_root(), "outputs", config.name, dfn.dir_name
#     )
#     os.makedirs(dir_abs_path, exist_ok=True)
#     if saver == "pickle":
#         file = os.path.join(dir_abs_path, dfn.file_name + ".pkl")
#         with open(file, "wb") as f:
#             pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
#     elif saver == "json":
#         file = os.path.join(dir_abs_path, dfn.file_name + ".json")
#         with open(file, "w") as f:
#             json.dump(data, f)
#     elif saver == "df_csv":
#         file = os.path.join(dir_abs_path, dfn.file_name + ".csv")
#         data.to_csv(file, index=False)
#     else:
#         logger.exception("Posible loader options: pickle, json")
#     logger.info(f"Data saved as:  {file}")


# def load_data(config, data_file_name, loader="pickle", dir_abs_path=None):
#     dfn = parse_data_file_name(data_file_name)
#     if dir_abs_path is None:
#         dir_abs_path = os.path.join(
#             get_project_root(), "outputs", config.name, dfn.dir_name
#         )
#     else:
#         dir_abs_path = os.path.join(dir_abs_path, dfn.dir_name)

#     if loader == "pickle":
#         file = os.path.join(dir_abs_path, dfn.file_name + ".pkl")
#         with open(file, "rb") as f:
#             data = pickle.load(f)
#     elif loader == "json":
#         file = os.path.join(dir_abs_path, dfn.file_name + ".json")
#         with open(file, "r") as f:
#             data = json.load(f)
#     elif loader == "df_csv":
#         file = os.path.join(dir_abs_path, dfn.file_name + ".csv")
#         data = pd.read_csv(file)
#     elif loader == "df_xlsx":
#         file = os.path.join(dir_abs_path, dfn.file_name + ".xlsx")
#         data = pd.read_excel(file)
#     else:
#         logger.exception("Posible loader options: pickle, json, df_csv, df_xlsx")

#     logger.info(f"Data loaded from:  {file}")
#     return data


def init_obj(module, module_name, module_args=None, *args, **kwargs):
    if module_args is None:
        module_args = {}
    assert all(
        [k not in module_args for k in kwargs]
    ), "Overwriting kwargs given in config file is not allowed"
    module_args.update(kwargs)
    return getattr(module, module_name)(*args, **module_args)


def init_ftn(module, module_name, module_args=None, *args, **kwargs):
    if module_args is None:
        module_args = {}
    assert all(
        [k not in module_args for k in kwargs]
    ), "Overwriting kwargs given in config file is not allowed"
    module_args.update(kwargs)
    return partial(getattr(module, module_name), *args, **module_args)


@dataclass
class Timer:
    name: str = field(default="")
    logger: logging.Logger = field(default=logger)

    def __post_init__(self):
        self.start = default_timer()
        self.end = None

    def stop(self):
        self.end = default_timer()
        self.logger.info(f"{self.name} took {self.end - self.start:.2f} seconds.")


def get_number_cpus(parallelize=-1):
    num_cpus = multiprocessing.cpu_count()
    if parallelize == -1:
        parallelize = num_cpus
    elif 1 <= parallelize <= num_cpus:
        pass
    elif parallelize > num_cpus:
        parallelize = num_cpus
        logger.warning(f"number of cpus changed from {num_cpus} to {parallelize}")
    else:
        raise ValueError("Define accurate number of cpus")
    return parallelize


def dict_zip(*dicts):
    if not dicts:
        return

    n = len(dicts[0])
    if any(len(d) != n for d in dicts):
        raise ValueError("arguments must have the same length")

    for key, first_val in dicts[0].items():
        yield key, first_val, *(other[key] for other in dicts[1:])


def equalize_dict_len(dict1, dict2):
    keys_both = set(list(dict1.keys()) + list(dict2.keys()))
    for key in keys_both:
        if key not in dict1:
            dict1[key] = None
        if key not in dict2:
            dict2[key] = None
    return dict1, dict2


def get_increasing_seq_indices(values_list):
    indices = []
    last_value = 0
    for idx, value in enumerate(values_list):
        if value > last_value:
            last_value = value
            indices.append(idx)
    return indices
