from types import SimpleNamespace

import matplotlib.pyplot as plt

from data_loader.data_loader import DataLoader
from initializer.cameras_corregistration import CamerasCorregistrator
from segmentator.segmentator import Segmentator
from utils import plot_utils, utils
from utils.image_utils import (average_signatures, filter_small_area_pixels,
                               limit_to_bounds)
from utils.plot_utils import display_images, pixels_select_lasso
from utils.utils import get_logger, load_data, save_data

logger = get_logger(name="show_image")

def main(cfg):
    data_loader = DataLoader(cfg).load_images()
    images_cam1 = data_loader.images.cam1
    images_cam2 = data_loader.images.cam2

    image_cam1 = images_cam1[cfg.image_idx]

    images = SimpleNamespace(cam1=image_cam1, cam2=None)

    if images_cam2:
        images.cam2 = images_cam2[cfg.image_idx]

    display_images(images)
    plt.show()
