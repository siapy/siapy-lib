from data_loader.data_loader import DataLoader
from initializer.cameras_corregistration import CamerasCorregistrator
from utils import plot_utils, utils

logger = utils.get_logger(name="select_signatures")

def main(cfg):
    data_loader = DataLoader(cfg)
    data_loader.change_dir("corregistrate").load_images()

    images_cam1 = data_loader.images.cam1
    images_cam2 = data_loader.images.cam2
    image_cam1 = images_cam1[cfg.image_idx]
    image_cam2 = images_cam2[cfg.image_idx]

    selected_pixels_cam1 = plot_utils.pixels_select_click(image_cam1)
    selected_pixels_cam2 = plot_utils.pixels_select_click(image_cam2)

    corregistrator = CamerasCorregistrator(cfg)
    corregistrator.align(selected_pixels_cam2, selected_pixels_cam1,
                         plot_progress=True, points_ordered=True)
    corregistrator.save_params()


	##### Save and load
    # selected_areas_cam1 = plot_utils.pixels_select_lasso(image_cam1)

    # selected_area = {
    #     "cam1": selected_areas_cam1[0].to_dict()
    # }
    # utils.save_data(selected_area, "lasso", cfg)

    # selected_pixels = utils.load_data(cfg, data_dir_name="lasso")
    # selected_areas_cam1 = [pd.DataFrame.from_dict(selected_pixels["cam1"])]
    #####





