import logging
import hydra
from omegaconf import DictConfig
import os


log = logging.getLogger(__name__)
@hydra.main(config_path="../configs", config_name="config")
def main_func(cfg: DictConfig):
    log.debug("Debug level message")
    log.info("Info level message")
    log.warning("Warning level message")

@hydra.main(config_path="../configs", config_name="config.yaml")
def get_dataset(cfg: DictConfig):
    name_of_dataset = cfg.dataset.name
    num_samples = cfg.num_samples

    if name_of_dataset == "dataset1":
        feature_size = cfg.dataset.feature_size
        return print(feature_size)

    elif name_of_dataset == "dataset2":
        dim1 = cfg.dataset.dim1
        dim2 = cfg.dataset.dim2
        return print(2)

    else:
        raise ValueError("You outplayed the developer")

if __name__ == "__main__":
    main_func()
    get_dataset()
