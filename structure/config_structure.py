from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Union

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class CameraConfig():
    name: str = MISSING
    model: str = MISSING
    image_display_brightness: Optional[float] = MISSING

@dataclass
class DataLoaderConfig:
    name: str = MISSING
    data_dir_path: str = MISSING
    path_ending_camera1: str = MISSING
    path_ending_camera2: str = MISSING
    corregistrate_dir_name: str = MISSING

@dataclass
class PreparatorConfig:
    labels_path_deliminator: str = MISSING
    labels_deliminator: str = MISSING
    image_slices_size_cam1: int = MISSING
    image_slices_size_cam2: int = MISSING
    percentage_of_background : int = MISSING
    reflectance_panel: Optional[float] = MISSING
    panel_filter_function: str = MISSING
    merge_images_by_specter: bool = MISSING
    match_labels_to_indices: bool = MISSING

@dataclass
class SegmentatorConfig:
    decision_function: str = MISSING
    classes: list = MISSING
    classes_keep: list = MISSING
    classes_remove: list = MISSING
    area_size_threshold_camera1: int = MISSING
    area_size_threshold_camera2: int = MISSING

@dataclass
class SelectorConfig:
    average: bool = MISSING
    item: Optional[str] = MISSING
    color: str = MISSING

@dataclass
class VisualiserConfig:
    camera1: bool = MISSING
    camera2: bool = MISSING
    images_indices: list = MISSING
    objects_indices: list = MISSING
    slices_indices: list = MISSING
    labels_names: list = MISSING
    iterate_over_images: bool = MISSING
    group_data_by: Optional[str] = MISSING
    groups: Optional[dict] = MISSING
    plot: Optional[dict] = MISSING

@dataclass
class Config:
    camera1: CameraConfig = MISSING
    camera2: CameraConfig = MISSING
    data_loader: DataLoaderConfig = MISSING
    preparator: PreparatorConfig = MISSING
    segmentator: SegmentatorConfig = MISSING
    selector: SelectorConfig = MISSING
    visualiser: VisualiserConfig = MISSING
    program: str = MISSING
    image_idx: int = MISSING
    name: str = MISSING


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(group="camera1", name="base_camera1", node=CameraConfig)
cs.store(group="camera2", name="base_camera2", node=CameraConfig)
cs.store(group="data_loader", name="base_data_loader", node=DataLoaderConfig)
cs.store(group="preparator", name="base_preparator", node=PreparatorConfig)
cs.store(group="segmentator", name="base_segmentator", node=SegmentatorConfig)
cs.store(group="selector", name="base_selector", node=SelectorConfig)
cs.store(group="visualiser", name="base_visualiser", node=VisualiserConfig)

