from siapy.utils.utils import get_logger

logger = get_logger(name="checkers")


def check_visualiser(config):
    temp_cfg = {"images_indices": config.images_indices, "objects_indices": config.objects_indices,
                "slices_indices": config.slices_indices, "labels_names": config.labels_names}
    options = ["__include_all__", "__include_only__", "__exclude_only__"]
    for key, val in temp_cfg.items():
        if val[0] not in options:
            msg = f"{key}:{val} not supported. Available options in visualiser.yaml: {options}"
            logger.error(msg)
            raise ValueError(msg)


def check_config(config):
    if "camera2" not in config:
        config.camera2 = None

    check_visualiser(config.visualiser)
    return config

