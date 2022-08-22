from omegaconf import OmegaConf

from siapy.utils.utils import get_logger

logger = get_logger(name="docs")

def display_help(cfg):
    logger.info(
        f"\n== Configuration groups == \n Compose your configuration \
from those groups (group=option): \n"
        )
    pass


def version(cfg):
    logger.info("\n\
Spectral imaging analysis for python (SiaPy). \n \
Version: v1.0.0 \n \
use: 'run --help' to see the documentation.  ")
