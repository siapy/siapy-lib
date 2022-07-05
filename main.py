import logging
import os

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from scripts import (corregistrate, perform_segmentation, prepare_data,
                     select_signatures, show_image, test_segmentation)

logger = logging.getLogger("main")


@hydra.main(config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))

    programs = {
        "show_image": show_image,
        "select_signatures": select_signatures,
        "corregistrate": corregistrate,
        "test_segmentation": test_segmentation,
        "perform_segmentation": perform_segmentation,
        "prepare_data": prepare_data,
    }
    program = programs[cfg.program]
    program.main(cfg)


if __name__ == "__main__":
    main()





