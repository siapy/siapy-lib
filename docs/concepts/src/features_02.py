import numpy as np
import pandas as pd

from siapy.features.spectral_indices import compute_spectral_indices

# Create sample spectral data
np.random.seed(42)
data = pd.DataFrame(
    {
        "R": np.random.random(100),
        "G": np.random.random(100),
    }
)

indices_df = compute_spectral_indices(
    data=data,
    spectral_indices=["BIXS", "RI"],  # Indices to compute
)
print(f"Computed indices\n: {indices_df.head()}")
