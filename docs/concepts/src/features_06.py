import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from siapy.entities import Pixels, Signatures, SpectralImage
from siapy.features import AutoSpectralIndicesClassification
from siapy.features.helpers import FeatureSelectorConfig
from siapy.features.spectral_indices import compute_spectral_indices, get_spectral_indices

# Create a mock spectral image with 4 bands (Red, Green, Blue, Near-infrared)
rng = np.random.default_rng(seed=42)
image_array = rng.random((50, 50, 4))  # height, width, bands (R, G, B, N)
image = SpectralImage.from_numpy(image_array)

# Define region of interest (ROI) pixels for sampling
roi_pixels = Pixels.from_iterable(
    [(10, 15), (12, 18), (15, 20), (18, 22), (20, 25), (25, 30), (28, 32), (30, 35), (32, 38), (35, 40)]
)

# Extract spectral signatures from ROI pixels
signatures = image.to_signatures(roi_pixels)
print(f"Extracted {len(signatures)} signatures from the image")

# Convert signatures to DataFrame and assign standard band names
spectral_data = signatures.signals.df.copy()
spectral_data = spectral_data.rename(columns=dict(zip(spectral_data.columns, ["R", "G", "B", "N"])))

# Create synthetic classification labels for demonstration purposes
_, y = make_classification(n_samples=len(spectral_data), n_features=4, random_state=42)
target = pd.Series(y[: len(spectral_data)])

# Get all spectral indices that can be computed with available bands
available_indices = get_spectral_indices(["R", "G", "B", "N"])
print(f"Found {len(available_indices)} computable spectral indices")

# Method 1: Manually compute spectral indices
indices_df = compute_spectral_indices(
    data=spectral_data,
    spectral_indices=list(available_indices.keys())[:10],  # Use first 10 indices
)
print(f"Computed {indices_df.shape[1]} spectral indices")

# Method 2: Automated feature selection with spectral indices
# Configure the feature selector
config = FeatureSelectorConfig(
    k_features=5,  # Select 5 best performing indices
    cv=3,  # Use 3-fold cross-validation
    verbose=0,
)

# Create automated selector that finds optimal spectral indices
auto_spectral = AutoSpectralIndicesClassification(
    spectral_indices=list(available_indices.keys())[:15],  # Use first 15 indices as candidates
    selector_config=config,
    merge_with_original=False,  # Return only selected indices, not original bands
)

# Apply feature selection to find the best spectral indices
selected_features = auto_spectral.fit_transform(spectral_data, target)
print(f"Selected {selected_features.shape[1]} optimal spectral indices")

# Create new signatures object with selected features
enhanced_signatures = Signatures.from_signals_and_pixels(signals=selected_features, pixels=signatures.pixels)
print(f"Created enhanced signatures with shape: {enhanced_signatures.signals.df.shape}")

# Display results - the enhanced signatures contain only the most informative spectral indices
print(f"Enhanced signatures DataFrame:\n{enhanced_signatures.to_dataframe().head()}")
