import glob
import logging
import multiprocessing
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
from funcy import log_durations

import segmentator.decision_algos as module
from utils.utils import (Timer, get_logger, get_project_root, init_obj,
                         load_data)

logger = get_logger(name="segmentator")

class Segmentator():
    def __init__(self, cfg):
        self.cfg = cfg
        self.algos = None

    def init_decision_function(self):
        cfg_arg = {'cfg': self.cfg}
        data = self._load_data()
        self.algos = SimpleNamespace(cam1=None, cam2=None)
        self.algos.cam1 = init_obj(module, self.cfg.misc.segmentator.decision_function, cfg_arg)
        self.algos.cam1.fit(data.cam1)

        if data.cam2:
            self.algos.cam2 = init_obj(module, self.cfg.misc.segmentator.decision_function, cfg_arg)
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

    @staticmethod
    def predict(algorithm, signatures, parallelize=0, num_cpus=1):
        if parallelize == 0:
            return list(map(algorithm.predict, signatures))
        if parallelize == -1:
            parallelize = num_cpus
        elif 1 <= parallelize <= num_cpus:
            pass
        else:
            raise ValueError('Parallelize should be in range [-1, number of cpus]')

        with multiprocessing.Pool(processes=parallelize) as p:
            targets = p.map(algorithm.predict, signatures)
        return targets

    def filter_areas(self, images, selected_areas):
        """_summary_

        Args:
            images (tuple): image od same scene for cam1 and cam2
            selected_areas (_type_): selected areas for cam1 and cam2
        """
        cls_remove = self.cfg.misc.segmentator.classes_remove
        parallelize = self.cfg.misc.segmentator.parallelize
        num_cpus = multiprocessing.cpu_count()

        selected_areas_out = SimpleNamespace(cam1=[], cam2=[])
        timer = Timer(name="Classification of signatures", logger=logger)
        for area_pix in selected_areas.cam1:
            signatures_df = images.cam1.to_signatures(area_pix)
            signatures = signatures_df.signature.to_list()
            targets = Segmentator.predict(self.algos.cam1, signatures,
                                          parallelize=parallelize,
                                          num_cpus=num_cpus)
            signatures_df["target"] = targets
            # remove rows with target in classes_remove
            signatures_df = signatures_df[~signatures_df.target.isin(cls_remove)].reset_index()
            selected_areas_out.cam1.append(signatures_df)

        # do the same for camera 2, if images are available
        if selected_areas.cam2:
            for area_pix in selected_areas.cam2:
                signatures_df = images.cam2.to_signatures(area_pix)
                signatures = signatures_df.signature.to_list()
                targets = Segmentator.predict(self.algos.cam2, signatures,
                                              parallelize=parallelize,
                                              num_cpus=num_cpus)
                signatures_df["target"] = targets
                # remove rows with target in classes_remove
                signatures_df = signatures_df[~signatures_df.target.isin(cls_remove)].reset_index()
                selected_areas_out.cam2.append(signatures_df)

        timer.stop()
        return selected_areas_out



