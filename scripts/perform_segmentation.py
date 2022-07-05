from types import SimpleNamespace

import matplotlib.pyplot as plt

from data_loader.data_loader import DataLoader
from corregistrator.corregistrator import Corregistrator
from segmentator.segmentator import Segmentator
from utils import plot_utils, utils
from utils.image_utils import average_signatures, limit_to_bounds
from utils.plot_utils import (display_images, pixels_select_lasso,
                              segmentation_buttons)
from utils.utils import get_logger, load_data, save_data

logger = get_logger(name="perform_segmentation")


def main(cfg):
    data_loader = DataLoader(cfg).load_images()
    corregistrator = Corregistrator(cfg).load_params()
    segmentator = Segmentator(cfg).init_decision_function()

    images_cam1 = data_loader.images.cam1
    images_cam2 = data_loader.images.cam2

    # check if images from camera 2 are loaded
    if not len(images_cam2):
        images_cam2 = [None]* len(images_cam1)

    image_idx = cfg.image_idx
    while 1:
        image_cam1 = images_cam1[image_idx]
        image_cam2 = images_cam2[image_idx]
        logger.info(f"_Processed index_: {image_idx}")
        logger.info(f"Processed files: \n -> camera1: {image_cam1.filename} \
                    \n -> camera2: {image_cam2.filename}")

        images = SimpleNamespace(cam1=image_cam1, cam2=image_cam2)

        # manually selection of object on the image of camera 1
        selected_areas_cam1 = pixels_select_lasso(image_cam1)
        images = SimpleNamespace(cam1=image_cam1, cam2=None)
        selected_areas = SimpleNamespace(cam1=selected_areas_cam1, cam2=None)

        # convert selected ares to camera 2 space
        if image_cam2 is not None:
            images.cam2 = image_cam2
            # tranform coordinates from cam1 to cam2
            selected_areas_cam2 = list(map(corregistrator.transform, selected_areas_cam1))
            # limit coordinates to the image size
            selected_areas.cam2 = list(map(limit_to_bounds(images.cam2.shape), selected_areas_cam2))

        # check whether first objet on image selected represents reference panel
        selected_area_panel_cam1 = None
        selected_area_panel_cam2 = None
        if cfg.preparator.reflectance_panel is not None:
            selected_area_panel_cam1 = selected_areas.cam1[0]
            selected_areas.cam1.pop(0)
            if selected_areas.cam2 is not None:
                selected_area_panel_cam2 = selected_areas.cam2[0]
                selected_areas.cam2.pop(0)

        selected_areas = segmentator.run(images, selected_areas)

        # append non segmented area of reference panel at the beginning of list
        if selected_area_panel_cam1 is not None:
            selected_areas.cam1.insert(0, selected_area_panel_cam1)
        if selected_area_panel_cam2 is not None:
            selected_areas.cam2.insert(0, selected_area_panel_cam2)

        # display and confirmation process
        display_images(images, selected_areas, colors=cfg.selector.color)
        flag = segmentation_buttons()

        if flag == "skip":
            image_idx += 1
        if flag == "repeat":
            continue
        if image_idx == len(images_cam1):
            break
        if flag == "save":
            segmentator.save_segmented(images, selected_areas, image_idx)
            image_idx += 1
