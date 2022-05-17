from data_loader import data_loader
from utils import utils

logger = utils.get_logger(name="show")

def show(cfg):
    dl = data_loader.DataLoader(cfg)
    dl.load_data()
