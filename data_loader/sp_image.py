import numpy as np
import pandas as pd


class SPImage():
    def __init__(self, sp_file, config):
        self._sp_file = sp_file
        self._image_arr = None
        self._brightness = config.image_display_brightness
        self.cfg = config

    def __repr__(self):
        return repr(self._sp_file)

    def to_display(self):
        db = self._sp_file.metadata["default bands"]
        db = list(map(int, db))
        image_3ch = self._sp_file.read_bands(db)
        image_3ch[:, :, 0] = image_3ch[:, :, 0] / (image_3ch[:, :, 0].max() / 255.) * self._brightness
        image_3ch[:, :, 1] = image_3ch[:, :, 1] / (image_3ch[:, :, 1].max() / 255.) * self._brightness
        image_3ch[:, :, 2] = image_3ch[:, :, 2] / (image_3ch[:, :, 2].max() / 255.) * self._brightness
        return image_3ch.astype("uint8")

    def to_numpy(self):
        if self._image_arr is None:
            self._image_arr = self._sp_file[:,:,:]
        return self._image_arr

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
    def shape(self):
        rows = self._sp_file.nrows
        samples = self._sp_file.ncols
        bands = self._sp_file.nbands
        return (rows, samples, bands)

    @property
    def filename(self):
        return self._sp_file.filename.split("\\")[-1].split(".")[0]

    @property
    def camera_name(self):
        return self.cfg.name

    @property
    def wavelengths(self):
        wavelen = self._sp_file.metadata["wavelength"]
        return list(map(float, wavelen))
