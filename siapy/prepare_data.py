from types import SimpleNamespace

from siapy.entities import DataLoader
from siapy.preparator import Preparator
from siapy.utils.utils import get_logger

logger = get_logger(name="prepare_data")


def main(cfg):
    data_loader = DataLoader(cfg)
    data_loader.change_dir("segmented_images").load_images()
    preparator = Preparator(cfg)

    # group slices of the same image into under same list
    images_cam1 = preparator.batch_images(data_loader.images.cam1)

    # check if images from camera 2 are loaded
    if cfg.camera2 is not None:
        images_cam2 = preparator.batch_images(data_loader.images.cam2)
    else:
        images_cam2 = [None] * len(images_cam1)

    for image_cam1, image_cam2 in zip(images_cam1, images_cam2):
        images_segmented = SimpleNamespace(cam1=image_cam1, cam2=image_cam2)
        preparator.run(images_segmented)
