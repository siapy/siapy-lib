import matplotlib.pyplot as plt

from data_loader.data_loader import DataLoader
from initializer.cameras_corregistration import CamerasCorregistrator
from segmentator.segmentator import Segmentator
from utils import plot_utils, utils
from utils.plot_utils import display_images
from utils.utils import get_logger, load_data, save_data

logger = get_logger(name="test_segmentation")

def main(cfg):
    data_loader = DataLoader(cfg).load_images()
    corregistrator = CamerasCorregistrator(cfg).load_params()
    segmentator = Segmentator(cfg).init_decision_function()

    images_cam1 = data_loader.images.cam1
    images_cam2 = data_loader.images.cam2

    image_cam1 = images_cam1[cfg.image_idx]
    images = {
        "cam1": image_cam1,
        "cam2": None
    }

    if images_cam2:
        images["cam2"] = images_cam2[cfg.image_idx]

