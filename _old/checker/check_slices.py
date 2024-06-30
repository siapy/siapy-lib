import pandas as pd

from siapy.entities.containers import SpectralImageContainer
from siapy.utils.utils import get_logger, load_data, parse_labels

logger = get_logger(name="check_slices")


def parse_indices(filename, labels_path_deliminator, labels_deliminator):
    indices = filename.split(labels_path_deliminator)[0].split(labels_deliminator)
    indices = list(map(int, indices))
    return indices


def get_number_of_slices(indices_and_filenames):
    slices_num = []
    images_indices_unique = indices_and_filenames.images_indices.unique()
    for image_idx in images_indices_unique:
        all_indices_img = indices_and_filenames[
            indices_and_filenames.images_indices == image_idx
        ]
        objects_indices_unique = all_indices_img.objects_indices.unique()
        for object_idx in objects_indices_unique:
            all_indices_obj = all_indices_img[
                all_indices_img.objects_indices == object_idx
            ].reset_index(drop=True)
            filepath = "__".join(all_indices_obj.filenames[0].split("__")[1:])
            all_indices_obj = all_indices_obj.dropna().reset_index(drop=True)
            slices_num.append([image_idx, object_idx, len(all_indices_obj), filepath])

    return pd.DataFrame(
        data=slices_num,
        columns=["images_indices", "objects_indices", "slices_len", "filepaths"],
    )


def create_output_msg_string(slices_num):
    msg = ""
    for image_idx in slices_num.images_indices.unique():
        slices_num_at_idx = slices_num[slices_num.images_indices == image_idx]
        msg += f"\nImage {image_idx}:  "
        msg += f"Filepath: {slices_num_at_idx.filepaths.iloc[0]} \n"
        msg += slices_num_at_idx[["objects_indices", "slices_len"]].to_string()
    return msg


def main(cfg):
    data_loader = SpectralImageContainer(cfg)
    data_loader.change_dir("converted_images").load_images()

    images_cam1 = data_loader.images.cam1
    images_cam2 = data_loader.images.cam2

    filenames_cam1 = [image.filename for image in images_cam1]
    filenames_cam2 = [image.filename for image in images_cam2]

    labels_pd = cfg.preparator.labels_path_deliminator
    labels_d = cfg.preparator.labels_deliminator

    indices_cam1 = [
        parse_indices(filename, labels_pd, labels_d) for filename in filenames_cam1
    ]
    indices_cam2 = [
        parse_indices(filename, labels_pd, labels_d) for filename in filenames_cam2
    ]

    indices_cam1 = pd.DataFrame(
        data=indices_cam1,
        columns=["images_indices", "objects_indices", "slices_indices"],
    )
    indices_cam2 = pd.DataFrame(
        data=indices_cam2,
        columns=["images_indices", "objects_indices", "slices_indices"],
    )

    filenames_cam1 = pd.DataFrame(data=filenames_cam1, columns=["filenames"])
    filenames_cam2 = pd.DataFrame(data=filenames_cam2, columns=["filenames"])

    indices_and_filenames_cam1 = pd.concat([indices_cam1, filenames_cam1], axis=1)
    indices_and_filenames_cam2 = pd.concat([indices_cam2, filenames_cam2], axis=1)

    slices_num_cam1 = get_number_of_slices(indices_and_filenames_cam1)
    slices_num_cam2 = get_number_of_slices(indices_and_filenames_cam2)

    msg_cam1 = create_output_msg_string(slices_num_cam1)
    msg_cam2 = create_output_msg_string(slices_num_cam2)

    msg = "Camera 1: \n" + msg_cam1 + "\n\nCamera 2: \n" + msg_cam2
    logger.info(f"Report: \n{msg}")

    if cfg.data_loader.labels_of_groups_file_name is not None:
        # load file with labels and groups
        labels_file = load_data(
            cfg,
            cfg.data_loader.labels_of_groups_file_name,
            loader="df_xlsx",
            dir_abs_path=cfg.data_loader.data_dir_path,
        )

        # extract labels from filepaths
        ref_panel = bool(cfg.preparator.reflectance_panel)
        # camera 1
        # extract labels from filepath
        labels_all = slices_num_cam1.apply(
            lambda row: row.filepaths.split("__")[0].split("_"), axis=1
        )
        slices_num_cam1["labels_all"] = labels_all
        slices_num_cam1["labels"] = slices_num_cam1.apply(
            lambda row: parse_labels(row, ref_panel), axis=1
        )
        # asign corresponding groups to labels
        slices_num_cam1 = slices_num_cam1.merge(labels_file, on="labels", how="left")

        # repeat the same as above for camera 2
        if not slices_num_cam2.empty:
            labels_all = slices_num_cam2.apply(
                lambda row: row.filepaths.split("__")[0].split("_"), axis=1
            )
            slices_num_cam2["labels_all"] = labels_all
            slices_num_cam2["labels"] = slices_num_cam2.apply(
                lambda row: parse_labels(row, ref_panel), axis=1
            )
            slices_num_cam2 = slices_num_cam2.merge(
                labels_file, on="labels", how="left"
            )

        # create report of number of slices per group
        # we suspect the same froups for cam1 and cam2
        # TODO: repeat the same for cam 2
        msg = ""
        groups_unique = sorted(slices_num_cam1.groups.unique())
        for group in groups_unique:
            slices_num_group = (
                slices_num_cam1[slices_num_cam1.groups == group]
                .sort_values(by="labels")
                .reset_index(drop=True)
            )
            sum_slices = slices_num_group.slices_len.sum()
            msg += f"\n Group: {group} \n"
            msg += slices_num_group[["labels", "slices_len"]].to_string()
            msg += f"\nTotal number of slices: {sum_slices}\n"
        logger.info(f"Report: \n{msg}")
