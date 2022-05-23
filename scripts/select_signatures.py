
import matplotlib.pyplot as plt

from data_loader.data_loader import DataLoader
from initializer.cameras_corregistration import CamerasCorregistrator
from utils import plot_utils, utils

logger = utils.get_logger(name="select_signatures")

def main(cfg):
    data_loader = DataLoader(cfg)
    corregistrator = CamerasCorregistrator(cfg)
    data_loader.load_images()
    corregistrator.load_params()

    images_cam1 = data_loader.images.cam1
    images_cam2 = data_loader.images.cam2
    image_cam1 = images_cam1[cfg.image_idx]
    image_cam2 = images_cam2[cfg.image_idx]


    selected_areas_cam1 = plot_utils.pixels_select_lasso(image_cam1)
    selected_areas_cam2 = list(map(corregistrator.transform, selected_areas_cam1))

    # selected_areas_cam2 = corregistrator.transform(selected_areas_cam1[0])
    # selected_areas_cam2 = [iPtsMov_t]

    images_display = [image_cam1, image_cam2]
    # images_selected_areas = [selected_areas_cam1]
    images_selected_areas = [selected_areas_cam1, selected_areas_cam2]
    colors = [["red", "blue", "blue", "blue"], ["red", "blue", "blue", "blue"]]

    plot_utils.display_images(images_display, images_selected_areas, colors)
    plt.show()
