import pandas as pd
from sklearn.datasets import make_classification

from siapy.features import AutoFeatClassification

# Generate sample data
X, y = make_classification(n_samples=200, n_features=5, random_state=42)
data = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
target = pd.Series(y)

# Create and configure AutoFeat
autofeat = AutoFeatClassification(
    random_seed=42,  # For reproducibility
    verbose=1,  # Show progress
)

# Fit and transform
features_engineered = autofeat.fit_transform(data, target)
print(f"Original features: {data.shape[1]}")
print(f"Engineered features: {features_engineered.shape[1]}")
