import spectral as sp
from tqdm import tqdm

from utils import utils

logger = utils.get_logger(name="data_loader")
# logger.info("Init")
# logger.warning("Warning level message")

class DataLoader():
    def __init__(self, cfg):
        self.cfg = cfg

    def load_data(self):
        logger.info("Init")
        pass
