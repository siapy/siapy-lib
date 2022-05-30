import glob
import os
from types import SimpleNamespace

import pandas as pd

import segmentator.decision_algos as module
from utils.utils import get_project_root, init_obj, load_data


class Segmentator():
    def __init__(self, cfg):
        self.cfg = cfg
        self.algos = None

    def init_decision_function(self):
        data = self._load_data()
        self.algos = SimpleNamespace(cam1=None, cam2=None)
        self.algos.cam1 = init_obj(module, self.cfg.misc.segmentator.decision_function)
        self.algos.cam1.fit(data.cam1)

        if data.cam2:
            self.algos.cam2 = init_obj(module, self.cfg.misc.segmentator.decision_function)
            self.algos.cam2.fit(data.cam2)
        return self

    def _load_data(self):
        # parses filenames from outputs/signatures folder
        def get_files_paths(label):
            project_root = get_project_root()
            paths = sorted(glob.glob(f"{project_root}/outputs/{self.cfg.name}/signatures/{label}/*"))
            # remove filenames extensions
            return [os.path.splitext(path)[0] for path in paths]

        classes_labels = self.cfg.misc.segmentator.classes
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

    def segment_image(self, images, selected_areas=None):


