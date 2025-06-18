import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from siapy.entities import Pixels, Signatures, SpectralImage
from siapy.features import AutoSpectralIndicesClassification
from siapy.features.helpers import FeatureSelectorConfig
from siapy.features.spectral_indices import compute_spectral_indices, get_spectral_indices

# Create mock spectral image with realistic spectral bands
rng = np.random.default_rng(seed=42)
image_array = rng.random((50, 50, 4))  # height, width, bands (R, G, B, N)
image = SpectralImage.from_numpy(image_array)

# Define region of interest pixels
roi_pixels = Pixels.from_iterable(
    [(10, 15), (12, 18), (15, 20), (18, 22), (20, 25), (25, 30), (28, 32), (30, 35), (32, 38), (35, 40)]
)

# Extract spectral signatures from the image
signatures = image.to_signatures(roi_pixels)
print(f"Extracted {len(signatures)} signatures from the image")

# Convert signatures to DataFrame with standard band names
spectral_data = signatures.signals.df.copy()
spectral_data = spectral_data.rename(columns=dict(zip(spectral_data.columns, ["R", "G", "B", "N"])))

# Create synthetic target labels for demonstration
_, y = make_classification(n_samples=len(spectral_data), n_features=4, random_state=42)
target = pd.Series(y[: len(spectral_data)])

# Compute available spectral indices for these bands
available_indices = get_spectral_indices(["R", "G", "B", "N"])
print(f"Found {len(available_indices)} computable spectral indices")

# 1.) Apply spectral indices computation directly
indices_df = compute_spectral_indices(
    data=spectral_data,
    spectral_indices=list(available_indices.keys())[:10],  # Use first 10 indices
)
print(f"Computed {indices_df.shape[1]} spectral indices")

# 2.) Use automated feature selection with spectral indices
config = FeatureSelectorConfig(
    k_features=5,  # Select 5 best indices
    cv=3,
    verbose=0,
)

auto_spectral = AutoSpectralIndicesClassification(
    spectral_indices=list(available_indices.keys())[:15],  # Use first 15 indices
    selector_config=config,
    merge_with_original=False,  # Only return selected indices
)

# Fit and transform the spectral data
selected_features = auto_spectral.fit_transform(spectral_data, target)
print(f"Selected {selected_features.shape[1]} optimal spectral indices")

# Create enhanced signatures with selected features
enhanced_signatures = Signatures.from_signals_and_pixels(signals=selected_features, pixels=signatures.pixels)
print(f"Created enhanced signatures with shape: {enhanced_signatures.signals.df.shape}")

# The enhanced signatures can now be used for further analysis
print(f"Enhanced signatures DataFrame:\n{enhanced_signatures.to_dataframe().head()}")
