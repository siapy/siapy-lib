import glob
import os
from types import SimpleNamespace

import spectral as sp
from rich.progress import track

from siapy.data_loader import SPImage
from siapy.utils.utils import get_logger, to_absolute_path

logger = get_logger(name="data_loader")

class DataLoader():
    def __init__(self, config):
        self._cfg = config
        self._paths = None
        self._images = None

        # internal variable to check if camera2 is present
        self.is_camera2 = False if config.camera2 is None else True

    def _load_images_paths(self, num_images):
        cfg_data_loader = self._cfg.data_loader
        paths_cam1 = sorted(glob.glob(os.path.join(cfg_data_loader.data_dir_path, "*" +
                                                      cfg_data_loader.path_ending_camera1)))
        # in case there are no images check mounted data directory inside docker container
        if not len(paths_cam1):
            paths_cam1 = sorted(glob.glob(os.path.join("/app/data", "*" +
                                                       cfg_data_loader.path_ending_camera1)))

        paths_cam1 = paths_cam1[:num_images] if num_images > 0 else paths_cam1
        paths_cam2 = None
        if self.is_camera2:
            paths_cam2 = sorted(glob.glob(os.path.join(cfg_data_loader.data_dir_path, "*" +
                                                        cfg_data_loader.path_ending_camera2)))
            # in case there are no images check mounted data directory inside docker container
            if not len(paths_cam2):
                paths_cam2 = sorted(glob.glob(os.path.join("/app/data", "*" +
                                                        cfg_data_loader.path_ending_camera2)))
            paths_cam2 = paths_cam2[:num_images] if num_images > 0 else paths_cam2
        paths = {
            "cam1": paths_cam1,
            "cam2": paths_cam2
        }
        return SimpleNamespace(**paths)

    def _import_spectral_images(self):
        cfg_cam1 = self._cfg.camera1
        images_cam1 = [SPImage(sp.envi.open(path), cfg_cam1)
                       for path in track(self._paths.cam1, "[green]Processing (camera1)...")]

        images_cam2 = None
        if self.is_camera2:
            cfg_cam2 = self._cfg.camera2
            images_cam2 = [SPImage(sp.envi.open(path), cfg_cam2)
                           for path in track(self._paths.cam2, "[green]Processing (camera2)...")]

        images = {
            "cam1": images_cam1,
            "cam2": images_cam2
        }
        return SimpleNamespace(**images)

    @property
    def images(self):
        return self._images

    def load_images(self, num_images=-1):
        self._paths = self._load_images_paths(num_images)
        self._images = self._import_spectral_images()
        logger.info("Spectral images loaded into memory")
        return self

    def change_dir(self, dir_name):
        cfg_data_loader = self._cfg.data_loader
        if dir_name == "corregistrate":
            data_dir_path = os.path.join(cfg_data_loader.data_dir_path,
                                        cfg_data_loader.corregistrate_dir_name)
        elif dir_name == "segmented_images":
            data_dir_path = to_absolute_path(f"outputs/{self._cfg.name}/images/segmented")
        elif dir_name == "converted_images":
            data_dir_path = to_absolute_path(f"outputs/{self._cfg.name}/images/converted")

        self._cfg.data_loader.data_dir_path = data_dir_path
        return self

