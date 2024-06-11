from types import SimpleNamespace

import matplotlib.pyplot as plt

from siapy.transformations import Corregistrator
from siapy.entities import DataLoader
from siapy.segmentator import Segmentator
from siapy.utils.image_utils import limit_to_bounds
from siapy.utils.plot_utils import display_images, pixels_select_lasso
from siapy.utils.utils import get_logger

logger = get_logger(name="test_segmentation")


def main(cfg):
    data_loader = DataLoader(cfg).load_images()
    segmentator = Segmentator(cfg).init_decision_function()

    # if only camera1, then corregistration is not needed
    if cfg.camera2 is not None:
        corregistrator = Corregistrator(cfg)
        corregistrator.load_params()

    images_cam1 = data_loader.images.cam1
    image_cam1 = images_cam1[cfg.image_idx]
    selected_areas_cam1 = pixels_select_lasso(image_cam1)

    # save_data(cfg, data={"ss": selected_areas_cam1},
    #                 data_file_name=f"random/selected_areas")
    # ld = load_data(cfg, data_file_name=f"random/selected_areas")
    # selected_areas_cam1 = ld["ss"]

    images = SimpleNamespace(cam1=image_cam1, cam2=None)
    selected_areas = SimpleNamespace(cam1=selected_areas_cam1, cam2=None)

    if cfg.camera2 is not None:
        images_cam2 = data_loader.images.cam2
        images.cam2 = images_cam2[cfg.image_idx]
        # tranform coordinates from cam1 to cam2
        selected_areas_cam2 = list(map(corregistrator.transform, selected_areas_cam1))
        # limit coordinates to the image size
        selected_areas.cam2 = list(
            map(limit_to_bounds(images.cam2.shape), selected_areas_cam2)
        )

    selected_areas = segmentator.run(images, selected_areas)

    display_images(images, selected_areas, colors=cfg.selector.color)
    plt.show()
