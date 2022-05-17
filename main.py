import logging
import os

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from scripts import show_image

logger = logging.getLogger("main")


@hydra.main(config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    if cfg.program == "show_image":
        show_image.show(cfg)


if __name__ == "__main__":
    main()





