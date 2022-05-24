import logging
import os

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from scripts import corregistrate, select_signatures, show_image

logger = logging.getLogger("main")


@hydra.main(config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))

    if cfg.program == "show_image":
        show_image.show(cfg)
    if cfg.program == "select_signatures":
        select_signatures.main(cfg)
    if cfg.program == "corregistrate":
        corregistrate.main(cfg)


if __name__ == "__main__":
    main()





