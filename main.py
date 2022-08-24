import logging
import os

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import open_dict

from siapy import (check_images, corregistrate, create_signatures,
                   perform_segmentation, prepare_data, select_signatures,
                   show_image, test_segmentation, visualise_signatures)
from siapy.utils.utils import get_logger
from structure import Config, __docs__, check_config

logger = logging.getLogger("main")


@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(cfg: Config) -> None:
    with open_dict(cfg):
        cfg = check_config(cfg)

    programs = {
        "show_image": show_image.main,
        "select_signatures": select_signatures.main,
        "corregistrate": corregistrate.main,
        "test_segmentation": test_segmentation.main,
        "perform_segmentation": perform_segmentation.main,
        "prepare_data": prepare_data.main,
        "create_signatures": create_signatures.main,
        "visualise_signatures": visualise_signatures.main,
        "check_images": check_images.main,
        "version": __docs__.version,
    }
    try:
        program = programs[cfg.program]
        program(cfg)
    except KeyError:
        logger.error(f"Command '{cfg.program}' not found.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()





