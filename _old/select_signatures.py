from types import SimpleNamespace

import matplotlib.pyplot as plt

from siapy.transformations import Corregistrator
from siapy.entities import DataLoader
from siapy.utils.image_utils import average_signatures, limit_to_bounds
from siapy.utils.plot_utils import display_images, pixels_select_lasso
from siapy.utils.utils import get_logger, save_data

logger = get_logger(name="select_signatures")


def main(cfg):
    data_loader = DataLoader(cfg)
    data_loader.load_images()

    # if only camera1, then corregistration is not needed
    if cfg.camera2 is not None:
        corregistrator = Corregistrator(cfg)
        corregistrator.load_params()

    images_cam1 = data_loader.images.cam1
    image_cam1 = images_cam1[cfg.image_idx]

    selected_areas_cam1 = pixels_select_lasso(image_cam1)

    # create images list to display
    # images = [image_cam1]
    images = SimpleNamespace(cam1=image_cam1, cam2=None)
    # create list of areas which will be shown on the images
    # selected_areas_display = [selected_areas_cam1]
    selected_areas = SimpleNamespace(cam1=selected_areas_cam1, cam2=None)
    # dictionary of signatures for each image
    selected_signatures = SimpleNamespace(
        cam1=list(map(image_cam1.to_signatures, selected_areas_cam1)), cam2=None
    )

    # perform if images from both cameras are available
    if cfg.camera2 is not None:
        images_cam2 = data_loader.images.cam2
        image_cam2 = images_cam2[cfg.image_idx]
        # tranform coordinates from cam1 to cam2
        selected_areas_cam2 = list(map(corregistrator.transform, selected_areas_cam1))
        # limit coordinates to the image size
        selected_areas_cam2 = list(
            map(limit_to_bounds(image_cam2.shape), selected_areas_cam2)
        )

        images.cam2 = image_cam2
        selected_areas.cam2 = selected_areas_cam2
        selected_signatures.cam2 = list(
            map(image_cam2.to_signatures, selected_areas_cam2)
        )

    # perform averaging of signatures per area selected
    if cfg.selector.average:
        selected_signatures.cam1 = list(
            map(average_signatures, selected_signatures.cam1)
        )
        if cfg.camera2 is not None:
            selected_signatures.cam2 = list(
                map(average_signatures, selected_signatures.cam2)
            )

    # save signatures
    save_data(
        cfg,
        data=selected_signatures,
        data_file_name=f"signatures/{cfg.selector.item}/img_{cfg.image_idx}",
    )

    # colors = [["red", "blue", "blue", "blue"], ["red", "blue", "blue", "blue"]]
    display_images(images, selected_areas, colors=cfg.selector.color)
    plt.show()
