from types import SimpleNamespace

import matplotlib.pyplot as plt

from siapy.entities import DataLoader
from siapy.utils.plot_utils import display_images
from siapy.utils.utils import get_logger

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
