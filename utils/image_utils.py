import os

import cv2
import hydra
import numpy as np
import pandas as pd
import spectral as sp

from utils.utils import get_logger, parse_data_file_name

logger = get_logger(name="image_utils")

def average_signatures(area_of_signatures):
    if area_of_signatures is not None:
        x_center = area_of_signatures.x.min() + (area_of_signatures.x.max()
                                            - area_of_signatures.x.min()) / 2
        y_center = area_of_signatures.y.min() + (area_of_signatures.y.max()
                                                - area_of_signatures.y.min()) / 2
        signatures_mean = [list(area_of_signatures.signature.mean())]

        data = {"x": int(x_center), "y": int(y_center), "signature": signatures_mean}
        return pd.DataFrame(data, columns=["x", "y", "signature"])
    else:
        return list()

def limit_to_bounds(image_shape):
    y_max = image_shape[0]
    x_max = image_shape[1]
    def _limit(points):
        points = points[(points.x >= 0) &
                        (points.y >= 0) &
                        (points.x < x_max) &
                        (points.y < y_max)]
        return points
    return _limit


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
            cv2.drawContours(image_thresh_copy, contours, num_contour, color=255, thickness=-1)
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

        return pd.DataFrame(pixels_coor_new.astype("int"), columns=["x","y","z"])

    return filter_area


def filter_with_decision_algo(image, algorithm, cls_remove):
    def filter_(area_pix):
        signatures_df = image.to_signatures(area_pix)
        signatures = signatures_df.signature.to_list()
        targets = list(map(algorithm.predict, signatures))
        signatures_df["target"] = targets
        # remove rows with target in classes_remove
        signatures_df = signatures_df[~signatures_df.target.isin(cls_remove)].reset_index()
        return signatures_df
    return filter_


def save_image(config, image, data_file_name, metadata=None):
    dfn = parse_data_file_name(data_file_name)
    dir_abs_path = hydra.utils.to_absolute_path(os.path.join("outputs", config.name, dfn.dir_name))
    os.makedirs(dir_abs_path, exist_ok=True)
    file = os.path.join(dir_abs_path, dfn.file_name + ".hdr")
    sp.envi.save_image(file, image, dtype=np.float32, metadata=metadata, force=True)
    logger.info(f"Images saved as:  {file}")



