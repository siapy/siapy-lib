import matplotlib.pyplot as plt

from data_loader.data_loader import DataLoader
from utils import plot_utils, utils

logger = utils.get_logger(name="select_signatures")

def select(cfg):
    data_loader = DataLoader(cfg)
    data_loader.change_dir("corregistrate").load_images()
    images_cam1 = data_loader.images.cam1
    images_cam2 = data_loader.images.cam2

    image_cam1 = images_cam1[cfg.image_idx]
    image_cam2 = images_cam2[cfg.image_idx]

    selected_pixels_cam1 = plot_utils.pixels_select_click(image_cam1)
    selected_pixels_cam2 = plot_utils.pixels_select_click(image_cam2)

