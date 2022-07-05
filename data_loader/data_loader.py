import glob
import logging
import os
from types import SimpleNamespace

import hydra
import numpy as np
import spectral as sp
from funcy import log_durations
from tqdm import tqdm

from data_loader.sp_image import SPImage
from utils.utils import get_logger

logger = get_logger(name="data_loader")

class DataLoader():
    def __init__(self, config):
        self._cfg = config
        self._paths = None
        self._images = None

    def _load_images_paths(self):
        cfg_data_loader = self._cfg.data_loader
        paths_cam1 = sorted(glob.glob(os.path.join(cfg_data_loader.data_dir_path, "*" +
                                                      cfg_data_loader.data_sources.path_ending_camera1)))
        paths_cam2 = sorted(glob.glob(os.path.join(cfg_data_loader.data_dir_path, "*" +
                                                      cfg_data_loader.data_sources.path_ending_camera2)))
        paths = {
            "cam1": paths_cam1,
            "cam2": paths_cam2
        }
        return SimpleNamespace(**paths)

    def _import_spectral_images(self):
        cfg_cam1 = self._cfg.camera1
        cfg_cam2 = self._cfg.camera2
        images_cam1 = [SPImage(sp.envi.open(path), cfg_cam1) for path in self._paths.cam1]
        images_cam2 = [SPImage(sp.envi.open(path), cfg_cam2) for path in self._paths.cam2]

        images = {
            "cam1": images_cam1,
            "cam2": images_cam2
        }
        return SimpleNamespace(**images)

    @property
    def images(self):
        return self._images

    def load_images(self):
        self._paths = self._load_images_paths()
        self._images = self._import_spectral_images()
        logger.info("Spectral images loaded into memory")
        return self

    def change_dir(self, dir_name):
        cfg_data_loader = self._cfg.data_loader
        if dir_name == "corregistrate":
            data_dir_path = os.path.join(cfg_data_loader.data_dir_path,
                                        cfg_data_loader.corregistrate_dir_name)
        elif dir_name == "segmented_images":
            data_dir_path = hydra.utils.to_absolute_path(f"outputs/{self._cfg.name}/images/segmented")

        self._cfg.data_loader.data_dir_path = data_dir_path
        return self

