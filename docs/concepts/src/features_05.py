import pandas as pd
from sklearn.datasets import make_classification

from siapy.features import AutoSpectralIndicesClassification
from siapy.features.helpers import FeatureSelectorConfig
from siapy.features.spectral_indices import get_spectral_indices

# Create spectral-like data
X, y = make_classification(n_samples=300, n_features=4, random_state=42)
data = pd.DataFrame(X, columns=["R", "G", "B", "N"])  # Red, Green, Blue, NIR
target = pd.Series(y)

# Get available spectral indices
available_indices = get_spectral_indices(["R", "G", "B", "N"])
print(f"Available indices: {len(available_indices)}")

# Configure feature selection
config = FeatureSelectorConfig(
    k_features=(5, 20),  # Select 5-20 best indices
    cv=5,  # Cross-validation for feature selection
    verbose=1,
)

# Create automated spectral indices classifier
auto_spectral = AutoSpectralIndicesClassification(
    spectral_indices=list(available_indices.keys()),
    selector_config=config,
    merge_with_original=True,  # Include original bands
)

# Fit and transform
enhanced_features = auto_spectral.fit_transform(data, target)
print(f"\nOriginal features: {data.shape[1]}")
print(f"Enhanced features: {enhanced_features.shape[1]}")
