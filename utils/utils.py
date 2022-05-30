import json
import logging
import os
import pickle
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import hydra


def get_logger(name, verbosity=2):
    logger = logging.getLogger(name)
    logger.setLevel(verbosity)
    return logger
logger = get_logger(name="utils")

def parse_data_file_name(data_file_name):
    dfn_split = data_file_name.split("/")
    dfn_file_name = dfn_split[-1]
    dfn_dir_name = "/".join(dfn_split[:-1])
    return SimpleNamespace(file_name=dfn_file_name, dir_name=dfn_dir_name)

def save_data(config, data, data_file_name, saver="pickle"):
    #TODO: change first subfolder to name of a project used
    """ Save data in json format

    Args:
        config (DictConfig): hydra config file
        data (dict): data stored in dictionary
        data_dir_name (str): name of directory where data will be stored
    """
    dfn = parse_data_file_name(data_file_name)
    dir_abs_path = hydra.utils.to_absolute_path(os.path.join("outputs", config.name, dfn.dir_name))
    os.makedirs(dir_abs_path, exist_ok=True)
    if saver == "pickle":
        file = os.path.join(dir_abs_path, dfn.file_name + ".pkl")
        with open(file, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif saver == "json":
        file = os.path.join(dir_abs_path, dfn.file_name + ".json")
        with open(file, 'w') as f:
            json.dump(data, f)
    else:
        logger.exception("Posible loader options: pickle, json")
    logger.info(f"Data saved to {file}")

def load_data(config, data_file_name, loader="pickle"):
    dfn = parse_data_file_name(data_file_name)
    dir_abs_path = hydra.utils.to_absolute_path(os.path.join("outputs", config.name, dfn.dir_name))
    if loader == "pickle":
        file = os.path.join(dir_abs_path, dfn.file_name + ".pkl")
        with open(file, 'rb') as f:
            data = pickle.load(f)
    elif loader == "json":
        file = os.path.join(dir_abs_path, dfn.file_name + ".json")
        with open(file, 'r') as f:
            data = json.load(f)
    else:
        logger.exception("Posible loader options: pickle, json")

    logger.info(f"Data loaded from {file}")
    return SimpleNamespace(**data)

def init_obj(module, module_name, module_args=None, *args, **kwargs):
    if module_args is None:
        module_args = {}
    assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
    module_args.update(kwargs)
    return getattr(module, module_name)(*args, **module_args)

def init_ftn(module, module_name, module_args=None, *args, **kwargs):
    if module_args is None:
        module_args = {}
    assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
    module_args.update(kwargs)
    return partial(getattr(module, module_name), *args, **module_args)

def get_project_root() -> Path:
    return Path(__file__).parent.parent




