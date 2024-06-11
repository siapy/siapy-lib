import glob
import os
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd

import siapy.segmentator.decision_algos as module
from siapy.core import logger
from siapy.utils.image_utils import save_image
from siapy.utils.utils import Timer, get_number_cpus, get_project_root, init_obj


class Segmentator:
    def __init__(self, config):
        self.cfg = config
        self.algos = None

    def init_decision_function(self):
        cfg_arg = {"cfg": self.cfg}
        data = self._load_data()
        self.algos = SimpleNamespace(cam1=None, cam2=None)
        self.algos.cam1 = init_obj(
            module, self.cfg.segmentator.decision_function, cfg_arg
        )
        self.algos.cam1.fit(data.cam1)

        if data.cam2:
            self.algos.cam2 = init_obj(
                module, self.cfg.segmentator.decision_function, cfg_arg
            )
            self.algos.cam2.fit(data.cam2)
        return self

    def _load_data(self):
        # parses filenames from outputs/signatures folder
        def get_files_paths(label):
            project_root = get_project_root()
            paths = sorted(
                glob.glob(
                    f"{project_root}/outputs/{self.cfg.name}/signatures/{label}/*"
                )
            )
            # remove filenames extensions
            return [os.path.splitext(path)[0] for path in paths]

        classes_labels = self.cfg.segmentator.classes
        # initiate data structure for data - for each camera for all avaiblable calsses
        data = SimpleNamespace(cam1={}, cam2={})

        # iterate all classes (over all subfolders in signatures folder)
        for label in classes_labels:
            sigs_merged_cam1 = []
            sigs_merged_cam2 = []
            # iterate over all files in particular subfoler
            for file_path in get_files_paths(label):
                loaded_signatures = load_data(self.cfg, file_path)
                # merge signatures from same file
                sigs_merged_cam1.append(pd.concat(loaded_signatures.cam1))
                if loaded_signatures.cam2:
                    sigs_merged_cam2.append(pd.concat(loaded_signatures.cam2))

            # merge signatures from all files
            data.cam1[label] = pd.concat(sigs_merged_cam1, ignore_index=True)
            if loaded_signatures.cam2:
                data.cam2[label] = pd.concat(sigs_merged_cam2, ignore_index=True)

        return data

    def run(self, images, selected_areas):
        """_summary_

        Args:
            images (tuple): image od same scene for cam1 and cam2
            selected_areas (_type_): selected areas for cam1 and cam2
        """
        cls_remove = self.cfg.segmentator.classes_remove
        area_thrs_cam1 = self.cfg.segmentator.area_size_threshold_camera1
        area_thrs_cam2 = self.cfg.segmentator.area_size_threshold_camera2
        n_jobs = get_number_cpus()

        selected_areas_out = SimpleNamespace(cam1=None, cam2=[])
        timer = Timer(name="Classification of signatures", logger=logger)

        # TODO: figure why this increases time for calculation
        # # change image to numpy array
        # images.cam1.to_numpy()
        # images.cam2.to_numpy()

        # define filter functions for each camera using decision algo
        filter_cam1 = Segmentator.filter_with_decision_algo(
            images.cam1, self.algos.cam1, cls_remove
        )
        filter_cam2 = Segmentator.filter_with_decision_algo(
            images.cam2, self.algos.cam2, cls_remove
        )

        # execution of filter functions
        area_pix = [filter_cam1(area_pix) for area_pix in selected_areas.cam1]

        # filter by size camera 1
        selected_areas_out.cam1 = list(
            map(
                Segmentator.filter_small_area_pixels(images.cam1, area_thrs_cam1),
                area_pix,
            )
        )

        # do the same for camera 2, if images are available
        if selected_areas.cam2:
            area_pix = [filter_cam2(area_pix) for area_pix in selected_areas.cam2]
            # filter by size camera 2
            selected_areas_out.cam2 = list(
                map(
                    Segmentator.filter_small_area_pixels(images.cam2, area_thrs_cam2),
                    area_pix,
                )
            )

        timer.stop()
        return selected_areas_out

    def save_segmented(self, images, selected_areas, image_idx):
        def _save_segmented_image(selected_areas, image, metadata, image_idx):
            image_arr = image.to_numpy()
            for area_idx, area in enumerate(selected_areas):
                x_max = area.x.max()
                x_min = area.x.min()
                y_max = area.y.max()
                y_min = area.y.min()
                # create new image
                image_arr_area = np.nan * np.ones(
                    (y_max - y_min + 1, x_max - x_min + 1, image.shape[2])
                )
                # convert original coordinates to coordinates for new image
                y_coor = area.y - y_min
                x_coor = area.x - x_min
                # write values from original image to new image
                image_arr_area[y_coor, x_coor, :] = image_arr[area.y, area.x, :]

                data_file_name = (
                    f"images/segmented/{image_idx}_{area_idx}__{image.filename}"
                )
                save_image(self.cfg, image_arr_area, data_file_name, metadata)

        image_cam1 = images.cam1
        selected_areas_cam1 = selected_areas.cam1
        metadata_cam1 = image_cam1.file.metadata
        _save_segmented_image(selected_areas_cam1, image_cam1, metadata_cam1, image_idx)

        if images.cam2 is not None:
            image_cam2 = images.cam2
            selected_areas_cam2 = selected_areas.cam2
            metadata_cam2 = image_cam2.file.metadata
            _save_segmented_image(
                selected_areas_cam2, image_cam2, metadata_cam2, image_idx
            )

    @staticmethod
    def filter_small_area_pixels(image, threshold_area_size):
        def filter_contours(contours, threshold):
            contours_filtered = []
            for contour in contours:
                if cv2.contourArea(contour) > threshold:
                    contours_filtered.append(contour)
            return contours_filtered

        def get_contours_area_pixels(image_thresh, contours):
            pixel_coor = []
            for num_contour in range(len(contours)):
                image_thresh_copy = np.zeros_like(image_thresh)
                cv2.drawContours(
                    image_thresh_copy, contours, num_contour, color=255, thickness=-1
                )
                pts = np.where(image_thresh_copy == 255)
                pixel_coor.append(list([pts[0], pts[1]]))
            return pixel_coor

        def filter_area(selected_area):
            y_dim, x_dim, _ = image.shape
            data_map = np.zeros((y_dim, x_dim), dtype="uint8")
            data_map[selected_area.y, selected_area.x] = 255

            contours, _ = cv2.findContours(
                data_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            contours = filter_contours(contours, threshold_area_size)
            pixels_coor = get_contours_area_pixels(data_map, contours)

            pixels_coor_new = np.empty([0, 3])
            for pixels in pixels_coor:
                coor_y = pixels[0]
                coor_x = pixels[1]
                coor_xy = np.column_stack((coor_x, coor_y, np.ones(len(coor_x))))
                pixels_coor_new = np.append(pixels_coor_new, coor_xy, axis=0)

            return pd.DataFrame(pixels_coor_new.astype("int"), columns=["x", "y", "z"])

        return filter_area

    @staticmethod
    def filter_with_decision_algo(image, algorithm, cls_remove):
        def filter_(area_pix):
            signatures_df = image.to_signatures(area_pix)
            signatures = signatures_df.signature.to_list()
            targets = algorithm.predict(signatures)
            signatures_df["target"] = targets
            # remove rows with target in classes_remove
            signatures_df = signatures_df[
                ~signatures_df.target.isin(cls_remove)
            ].reset_index()
            return signatures_df

        return filter_
