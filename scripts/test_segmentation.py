from types import SimpleNamespace

import matplotlib.pyplot as plt

from data_loader.data_loader import DataLoader
from initializer.cameras_corregistration import CamerasCorregistrator
from segmentator.segmentator import Segmentator
from utils import plot_utils, utils
from utils.image_utils import average_signatures, limit_to_bounds
from utils.plot_utils import display_images, pixels_select_lasso
from utils.utils import get_logger, load_data, save_data

logger = get_logger(name="test_segmentation")

def main(cfg):
    data_loader = DataLoader(cfg).load_images()
    corregistrator = CamerasCorregistrator(cfg).load_params()
    segmentator = Segmentator(cfg).init_decision_function()

    images_cam1 = data_loader.images.cam1
    images_cam2 = data_loader.images.cam2

    image_cam1 = images_cam1[cfg.image_idx]
    # selected_areas_cam1 = pixels_select_lasso(image_cam1)

    # save_data(cfg, data={"ss": selected_areas_cam1},
                    # data_file_name=f"random/selected_areas")
    ld = load_data(cfg, data_file_name=f"random/selected_areas")
    selected_areas_cam1 = ld["ss"]

    images = SimpleNamespace(cam1=image_cam1, cam2=None)
    selected_areas = SimpleNamespace(cam1=selected_areas_cam1, cam2=None)

    if images_cam2:
        images.cam2 = images_cam2[cfg.image_idx]
        # tranform coordinates from cam1 to cam2
        selected_areas_cam2 = list(map(corregistrator.transform, selected_areas_cam1))
        # limit coordinates to the image size
        selected_areas.cam2 = list(map(limit_to_bounds(images.cam2.shape), selected_areas_cam2))

    selected_areas = segmentator.filter_areas(images, selected_areas)


    display_images(images, selected_areas, colors=cfg.misc.selector.color)
    plt.show()


