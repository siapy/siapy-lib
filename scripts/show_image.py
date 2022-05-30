import matplotlib.pyplot as plt

from data_loader.data_loader import DataLoader
from utils import plot_utils, utils

logger = utils.get_logger(name="show")

def show(cfg):
    data_loader = DataLoader(cfg)
    data_loader.load_images()

    images_cam1 = data_loader.images.cam1
    images_cam2 = data_loader.images.cam2

    image_cam1 = images_cam1[cfg.image_idx]
    images_display = [image_cam1]

    if images_cam2:
        image_cam2 = images_cam2[cfg.image_idx]
        images_display.append(image_cam2)

    plot_utils.display_images(images_display)
    plt.show()

