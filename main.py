import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from scripts import *

log = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    # log.debug("Debug level message")
    # log.info("Info level message")
    # log.warning("Warning level message")

    if cfg.program == "show_image":
        show_image.show(cfg)


if __name__ == "__main__":
    main()

