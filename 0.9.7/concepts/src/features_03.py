import numpy as np
import pandas as pd

from siapy.features.spectral_indices import compute_spectral_indices

# --8<-- [start:map]
# Data with custom column names
custom_data = pd.DataFrame(
    {"red_band": np.random.random(100), "green_band": np.random.random(100), "nir_band": np.random.random(100)}
)

# Map custom names to standard band acronyms
bands_map = {"red_band": "R", "green_band": "G", "nir_band": "N"}

indices_df = compute_spectral_indices(data=custom_data, spectral_indices=["NDVI", "GNDVI"], bands_map=bands_map)

# --8<-- [end:map]
print(f"Computed indices with custom mapping:\n{indices_df.head()}")
