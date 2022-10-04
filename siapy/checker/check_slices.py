import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from siapy.data_loader.data_loader import DataLoader
from siapy.utils.utils import get_logger


logger = get_logger(name="check_slices")

def parse_labels(filename, labels_path_deliminator, labels_deliminator):
    indices = filename.split(labels_path_deliminator)[0].split(labels_deliminator)
    indices = list(map(int, indices))
    return indices

def get_number_of_slices(indices):
    slices_num = []
    images_indices_unique = indices.images_indices.unique()
    for image_idx in images_indices_unique:
        all_indices_img = indices[indices.images_indices == image_idx]
        objects_indices_unique = all_indices_img.objects_indices.unique()
        for object_idx in objects_indices_unique:
            all_indices_obj = all_indices_img[all_indices_img.objects_indices == object_idx]
            slices_num.append([image_idx, object_idx, len(all_indices_obj)])

    return pd.DataFrame(data=slices_num, columns=["images_indices", "objects_indices", "slices_len"])

def create_output_msg_string(slices_num):
    msg = ""
    for image_idx in slices_num.images_indices.unique():
        msg += f"\n Image {image_idx}: \n"
        msg += slices_num[slices_num.images_indices == image_idx] \
                    [["objects_indices", "slices_len"]].to_string()
    return msg

def main(cfg):
    data_loader = DataLoader(cfg)
    data_loader.change_dir("converted_images").load_images()

    images_cam1 = data_loader.images.cam1
    images_cam2 = data_loader.images.cam2

    filenames_cam1 = [image.filename for image in images_cam1]
    filenames_cam2 = [image.filename for image in images_cam2]

    labels_pd = cfg.preparator.labels_path_deliminator
    labels_d = cfg.preparator.labels_deliminator

    indices_cam1 = [parse_labels(filename, labels_pd, labels_d) for filename in filenames_cam1]
    indices_cam2 = [parse_labels(filename, labels_pd, labels_d) for filename in filenames_cam2]

    indices_cam1 = pd.DataFrame(data=indices_cam1,
                                columns=["images_indices", "objects_indices", "slices_indices"])
    indices_cam2 = pd.DataFrame(data=indices_cam2,
                                columns=["images_indices", "objects_indices", "slices_indices"])

    slices_num_cam1 = get_number_of_slices(indices_cam1)
    slices_num_cam2 = get_number_of_slices(indices_cam2)

    msg_cam1 = create_output_msg_string(slices_num_cam1)
    msg_cam2 = create_output_msg_string(slices_num_cam2)

    msg = "Camera 1: \n" + msg_cam1 + "\n\nCamera 2: \n" + msg_cam2
    logger.info(f"Report: \n{msg}")

    plt.hist(slices_num_cam1.slices_len.to_numpy())
    plt.show()

