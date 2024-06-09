import numpy as np

from siapy.entities import DataLoader
from siapy.utils.plot_utils import plot_signatures
from siapy.utils.utils import (
    get_increasing_seq_indices,
    get_logger,
    load_data,
    save_data,
)

logger = get_logger(name="visualise_signatures")


def modify_dataframe(data, col_name, instructions):
    instruction = instructions[0]
    instructions.pop(0)
    # convert to strings even if integer values written
    instructions = list(map(str, instructions))
    if len(instructions):
        if instruction == "__include_all__":
            pass
        elif instruction == "__exclude_only__":
            data = data[~data[col_name].isin(instructions)]
        elif instruction == "__include_only__":
            data = data[data[col_name].isin(instructions)]
    else:
        logger.warning(f"No instructions for {col_name}")
    return data.reset_index(drop=True)


def append_groups_to_dataframe(data, groups, group_data_by):
    # create new column called groups
    data = data.assign(groups=np.nan)
    if group_data_by in data.columns:
        for group_name, group_list in groups.items():
            # in case in list integer values, convert to strings
            group_list = list(map(str, group_list))
            # get bolean indices indicating where groups applies
            group_indices = data[group_data_by].isin(group_list)
            # set group name
            data.loc[group_indices, "groups"] = group_name
    else:
        logger.warning(f"Data groups not set by: <{group_data_by}>")
    return data


def merge_cameras_rows(cfg, data):
    cam1_name = cfg.camera1.name
    cam2_name = cfg.camera2.name
    data_cam1 = data[data.camera_names == cam1_name]
    data_cam2 = data[data.camera_names == cam2_name]

    # remove camera_names column
    data_cam1 = data_cam1.drop(["camera_names"], axis=1)
    data_cam2 = data_cam2.drop(["camera_names"], axis=1)

    # check if dataframe not enabled create empty columns
    if not cfg.visualiser.camera1:
        data_cam1 = data_cam2.copy()
        data_cam1[["signatures", "filenames"]] = np.nan
    if not cfg.visualiser.camera2:
        data_cam2 = data_cam1.copy()
        data_cam2[["signatures", "filenames"]] = np.nan

    # rename colunms of each dataframe and merge
    data_cam1 = data_cam1.rename(
        columns={
            "signatures": "signatures_camera1",
            "filenames": "filenames_camera1",
        }
    )
    data_cam2 = data_cam2.rename(
        columns={
            "signatures": "signatures_camera2",
            "filenames": "filenames_camera2",
        }
    )
    data = data_cam1.merge(
        data_cam2,
        on=[
            "images_indices",
            "objects_indices",
            "slices_indices",
            "labels_all",
            "labels_names",
            "groups",
        ],
    )
    # create signature_megred column
    data["signatures_merged"] = np.nan
    return data


def merge_signatures_and_evaluate(row):
    if str(row.signatures_camera1) == "nan":
        row.signatures_merged = eval(row.signatures_camera2)
    elif str(row.signatures_camera2) == "nan":
        row.signatures_merged = eval(row.signatures_camera1)
    else:
        row.signatures_merged = eval(row.signatures_camera1) + eval(
            row.signatures_camera2
        )
    return row


def evaluate_signatures(row):
    row.signatures_merged = eval(row.signatures_merged)
    return row


def filter_by_indices(row, indices_inc):
    if indices_inc is not None:
        row.signatures_merged = list(np.array(row.signatures_merged)[indices_inc])
    return row


def main(cfg):
    data = load_data(cfg, data_file_name="files/data", loader="df_csv")
    config_vis = cfg.visualiser

    # remove or leave data from each camera
    if not config_vis.camera1:
        data = data[~(data["camera_names"] == cfg.camera1.name)]
    if not config_vis.camera2:
        data = data[~(data["camera_names"] == cfg.camera2.name)]
    if not (config_vis.camera1 or config_vis.camera2):
        logger.error("No camera selected for visualisation.")
        return

    # modify df based on config
    temp_cfg = {
        "images_indices": config_vis.images_indices,
        "objects_indices": config_vis.objects_indices,
        "slices_indices": config_vis.slices_indices,
        "labels_names": config_vis.labels_names,
    }

    for col_name, instructions in temp_cfg.items():
        data = modify_dataframe(data, col_name, instructions)

    # define groups of which signatures are a part of
    groups = config_vis.groups
    group_data_by = config_vis.group_data_by
    data = append_groups_to_dataframe(data, groups, group_data_by)

    # further modify data for plotting purpose
    # use dataloader to extract wavelengths
    data_loader = DataLoader(cfg)
    data_loader.change_dir("converted_images").load_images(1)

    # get wavelengths
    wavelengths_cam1 = None
    wavelengths_cam2 = None
    if data_loader.images.cam1 is not None and len(data_loader.images.cam1):
        wavelengths_cam1 = data_loader.images.cam1[0].wavelengths
    if data_loader.images.cam2 is not None and len(data_loader.images.cam2):
        wavelengths_cam2 = data_loader.images.cam2[0].wavelengths

    # merge wavelengths from both cameras
    x_scat = []
    if config_vis.plot.x_wavelengths:
        if config_vis.camera1 and wavelengths_cam1 is not None:
            x_scat += wavelengths_cam1
        if config_vis.camera2 and wavelengths_cam2 is not None:
            x_scat += wavelengths_cam2

        # find increasing sequences in wavelengths list
        if len(x_scat):
            indices_inc = get_increasing_seq_indices(x_scat)
            x_scat = np.array(x_scat)[indices_inc]
        else:
            indices_inc = None
            x_scat = None

    # check whether images were merged in preparation proces
    if not cfg.preparator.merge_images_by_specter:
        # merge dataframe and signatures from both cameras
        data = merge_cameras_rows(cfg, data)
        # merge signatures from both cameras
        data = data.apply(lambda row: merge_signatures_and_evaluate(row), axis=1)

    else:
        data = data.rename(columns={"signatures": "signatures_merged"})
        data = data.apply(lambda row: evaluate_signatures(row), axis=1)
    # filter by wavelengts
    data = data.apply(lambda row: filter_by_indices(row, indices_inc), axis=1)

    # save modified data
    save_data(cfg, data=data, data_file_name="files/data_parsed", saver="df_csv")

    # if iteration over seperate image enabled
    if config_vis.iterate_over_images:
        for image_idx in data.images_indices.unique():
            # get data part coresponding to particular image
            data_slice = data[data.images_indices == image_idx]
            plot_df = str(
                data_slice[
                    [
                        "images_indices",
                        "objects_indices",
                        "slices_indices",
                        "labels_names",
                        "groups",
                    ]
                ]
            )
            logger.info(f"Data plotted: \n {plot_df}")
            plot_signatures(
                data_slice.signatures_merged.tolist(),
                data_slice.objects_indices.tolist(),
                x_scat=x_scat,
            )
    else:
        signatures = data.signatures_merged.to_list()
        groups_labels = data.groups.to_list()
        plot_df = str(
            data[
                [
                    "images_indices",
                    "objects_indices",
                    "slices_indices",
                    "labels_names",
                    "groups",
                ]
            ]
        )
        logger.info(f"Data plotted: \n {plot_df}")
        plot_signatures(signatures, groups_labels, x_scat=x_scat)
