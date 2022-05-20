

class SPImage():
    def __init__(self, sp_file, config):
        self._sp_file = sp_file
        self._cfg = config
        self._np_array = None

    def __repr__(self):
        return repr(self._sp_file)

    def to_display(self):
        db = self._sp_file.metadata["default bands"]
        db = list(map(int, db))
        image_3ch = self._sp_file.read_bands(db)
        image_3ch[:, :, 0] = image_3ch[:, :, 0] / (image_3ch[:, :, 0].max() / 255.)
        image_3ch[:, :, 1] = image_3ch[:, :, 1] / (image_3ch[:, :, 1].max() / 255.)
        image_3ch[:, :, 2] = image_3ch[:, :, 2] / (image_3ch[:, :, 2].max() / 255.)
        return image_3ch.astype("uint16")

    def to_numpy(self):
        if self._np_array is None:
            self._np_array = self._sp_file[:,:,:]
        return self._np_array

    @property
    def file(self):
        return self._sp_file



