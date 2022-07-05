from types import SimpleNamespace

import matplotlib.pyplot as plt

from data_loader.data_loader import DataLoader
from preparator.preparator import Preparator
from utils.plot_utils import display_images
from utils.utils import get_logger

logger = get_logger(name="prepare_data")

def main(cfg):
    data_loader = DataLoader(cfg)
    data_loader.change_dir("segmented_images").load_images()
    preparator = Preparator(cfg)

    # group slices of the same image into under same list
    images_cam1 = preparator.batch_images(data_loader.images.cam1)
    images_cam2 = preparator.batch_images(data_loader.images.cam2)

    # check if images from camera 2 are loaded
    if not len(images_cam2):
        images_cam2 = [None]* len(images_cam1)

    for image_cam1, image_cam2 in zip(images_cam1, images_cam2):
        images_segmented = SimpleNamespace(cam1=image_cam1, cam2=image_cam2)
        preparator.run(images_segmented)



