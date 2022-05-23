import json
import logging
import os

import hydra


def get_logger(name, verbosity=2):
    logger = logging.getLogger(name)
    logger.setLevel(verbosity)
    return logger

def save_data(config, data, data_dir_name):
    #TODO: revomve cfg from arguments
    dir_abs_path = hydra.utils.to_absolute_path(os.path.join("outputs", data_dir_name))
    os.makedirs(dir_abs_path, exist_ok=True)
    file = os.path.join(dir_abs_path, config.name + ".json")
    with open(file, 'w') as f:
        json.dump(data, f)

def load_data(config, data_dir_name):
    dir_abs_path = hydra.utils.to_absolute_path(os.path.join("outputs", data_dir_name))
    file = os.path.join(dir_abs_path, config.name + ".json")
    with open(file, 'r') as f:
        data = json.load(f)
    return data



