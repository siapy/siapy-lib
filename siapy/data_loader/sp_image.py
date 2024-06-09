import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import spectral as sp


@dataclass
class SPImage:
    def __init__(self, sp_file):
        self._sp_file = sp_file

    def __repr__(self):
        return repr(self._sp_file)

    @classmethod
    def envi_open(cls, hdr_path, img_path=None):
        sp_file = sp.envi.open(file=hdr_path, image=img_path)
        return cls(sp_file)

    def _remove_nan(self, image, nan_value=0):
        image_mask = np.bitwise_not(np.bool_(np.isnan(image).sum(axis=2)))
        image[~image_mask] = nan_value
        return image

    def to_display(self, brightness=1):
        db = self.metadata["default bands"]
        db = list(map(int, db))
        image_3ch = self._sp_file.read_bands(db)
        image_3ch = self._remove_nan(image_3ch, nan_value=0)
        image_3ch[:, :, 0] = (
            image_3ch[:, :, 0] / (image_3ch[:, :, 0].max() / 255.0) * brightness
        )
        image_3ch[:, :, 1] = (
            image_3ch[:, :, 1] / (image_3ch[:, :, 1].max() / 255.0) * brightness
        )
        image_3ch[:, :, 2] = (
            image_3ch[:, :, 2] / (image_3ch[:, :, 2].max() / 255.0) * brightness
        )
        return image_3ch.astype("uint8")

    def to_numpy(self, nan_value=None):
        image = self._sp_file[:, :, :]
        if nan_value is not None:
            image = self._remove_nan(image, nan_value)
        return image

    def to_signatures(self, pixels_loc):
        image_arr = self.to_numpy()
        signatures = image_arr[pixels_loc.y, pixels_loc.x, :]
        data = {"x": pixels_loc.x, "y": pixels_loc.y, "signature": list(signatures)}
        return pd.DataFrame(data, columns=["x", "y", "signature"])

    def mean(self, axis=None):
        image_arr = self.to_numpy()
        return np.nanmean(image_arr, axis=axis)

    @property
    def file(self):
        return self._sp_file

    @property
    def metadata(self):
        return self._sp_file.metadata

    @property
    def shape(self):
        rows = self._sp_file.nrows
        samples = self._sp_file.ncols
        bands = self._sp_file.nbands
        return (rows, samples, bands)

    @property
    def rows(self):
        return self._sp_file.nrows

    @property
    def cols(self):
        return self._sp_file.ncols

    @property
    def bands(self):
        return self._sp_file.nbands

    @property
    def filename(self):
        return self._sp_file.filename.split(os.sep)[-1].split(".")[0]

    @property
    def wavelengths(self):
        wavelen = self._sp_file.metadata["wavelength"]
        return list(map(float, wavelen))
