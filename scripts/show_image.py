import matplotlib.pyplot as plt

from data_loader.data_loader import DataLoader
from utils import plot_utils, utils

logger = utils.get_logger(name="show")

def show(cfg):
    data_loader = DataLoader(cfg)
    images_cam1 = data_loader.images.cam1
    images_cam2 = data_loader.images.cam2

    image_cam1 = images_cam1[cfg.image_idx]
    image_cam2 = images_cam2[cfg.image_idx]

    # out = plot_utils.pixels_select_click(image_display)
    selected_areas_cam1 = plot_utils.pixels_select_lasso(image_cam1)

    images_display = [image_cam1, image_cam2]
    images_selected_areas = [selected_areas_cam1]
    colors = [["red", "blue", "blue", "blue"]]

    plot_utils.display_images(images_display, images_selected_areas, colors)
    plt.show()

