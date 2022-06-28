from types import SimpleNamespace

import matplotlib.pyplot as plt

from data_loader.data_loader import DataLoader
from initializer.cameras_corregistration import CamerasCorregistrator
from segmentator.segmentator import Segmentator
from utils import plot_utils, utils
from utils.image_utils import average_signatures, limit_to_bounds
from utils.plot_utils import display_images, pixels_select_lasso, segmentation_buttons
from utils.utils import get_logger, load_data, save_data

logger = get_logger(name="perform_segmentation")

def get_filtered_selected_areas(image_cam1, image_cam2, segmentator, corregistrator):
    selected_areas_cam1 = pixels_select_lasso(image_cam1)
    images = SimpleNamespace(cam1=image_cam1, cam2=None)
    selected_areas = SimpleNamespace(cam1=selected_areas_cam1, cam2=None)

    if image_cam2 is not None:
        images.cam2 = image_cam2
        # tranform coordinates from cam1 to cam2
        selected_areas_cam2 = list(map(corregistrator.transform, selected_areas_cam1))
        # limit coordinates to the image size
        selected_areas.cam2 = list(map(limit_to_bounds(images.cam2.shape), selected_areas_cam2))

    selected_areas = segmentator.filter_areas(images, selected_areas)
    return selected_areas

# def save_image_objects(cfg, images, selected_areas):



def main(cfg):
    data_loader = DataLoader(cfg).load_images()
    corregistrator = CamerasCorregistrator(cfg).load_params()
    segmentator = Segmentator(cfg).init_decision_function()

    images_cam1 = data_loader.images.cam1
    images_cam2 = data_loader.images.cam2

    # check if images form camera 2 are loaded
    if not len(images_cam2):
        images_cam2 = [None]* len(images_cam1)

    idx = cfg.image_idx
    while 1:
        image_cam1 = images_cam1[idx]
        image_cam2 = images_cam2[idx]

        images = SimpleNamespace(cam1=image_cam1, cam2=image_cam2)
        selected_areas = get_filtered_selected_areas(image_cam1, image_cam2, segmentator, corregistrator)

        display_images(images, selected_areas, colors=cfg.misc.selector.color)
        flag = segmentation_buttons()

        if flag == "skip":
            idx += 1
        if flag == "repeat":
            continue
        if idx == len(images_cam1):
            break
        if flag == "save":
            save_image_objects(cfg, images, selected_areas)
