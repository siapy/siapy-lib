from pathlib import Path

import numpy as np
import spectral as sp


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)
    return dirname


def save_image(file_name, image, metadata=None):
    sp.envi.save_image(
        file_name, image, dtype=np.float32, metadata=metadata, force=True
    )
