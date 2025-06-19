import numpy as np

from siapy.datasets.schemas import TabularDatasetData
from siapy.utils.plots import display_signals

# Create sample spectral signatures dataset
rng = np.random.default_rng(seed=42)

# Generate synthetic spectral data (simulate 4 bands: R, G, B, NIR)
n_samples = 50
n_bands = 4
spectral_data = rng.normal(0.5, 0.2, (n_samples, n_bands))
spectral_data = np.clip(spectral_data, 0, 1)  # Ensure valid reflectance values

# Create two classes with different spectral characteristics
class_a_indices = np.arange(0, 25)
class_b_indices = np.arange(25, 50)

# Modify spectral characteristics for each class
spectral_data[class_a_indices, 3] += 0.3  # Higher NIR for vegetation-like class
spectral_data[class_b_indices, 0] += 0.2  # Higher red for soil-like class

# Create dataset structure
data_dict = {
    "pixels": {
        "x": list(range(n_samples)),
        "y": [0] * n_samples,  # All from same row for simplicity
    },
    "signals": {f"band_{i}": spectral_data[:, i].tolist() for i in range(n_bands)},
    "metadata": {
        "sample_id": [f"sample_{i}" for i in range(n_samples)],
    },
    "target": {
        "label": ["vegetation"] * 25 + ["soil"] * 25,
        "value": [0] * 25 + [1] * 25,
        "encoding": {"vegetation": 0, "soil": 1},
    },
}

# Create TabularDatasetData object
dataset = TabularDatasetData.from_dict(data_dict)

# Basic signal plotting
print("Displaying basic spectral signals...")
display_signals(dataset)

# Customized signal plotting
print("Displaying customized spectral signals...")
display_signals(
    dataset,
    figsize=(10, 6),
    dpi=100,
    colormap="plasma",
    x_label="Spectral Bands (R, G, B, NIR)",
    y_label="Reflectance",
    label_fontsize=16,
    tick_params_label_size=14,
    legend_fontsize=12,
    legend_frameon=True,
)

# Additional example with more classes
print("Creating dataset with multiple classes...")

# Extend dataset to include a third class
extended_data = data_dict.copy()
extended_data["pixels"]["x"].extend(list(range(50, 75)))
extended_data["pixels"]["y"].extend([0] * 25)

# Add water-like spectral characteristics (low NIR, moderate blue)
water_spectra = rng.normal(0.3, 0.1, (25, n_bands))
water_spectra[:, 1] += 0.2  # Higher blue
water_spectra[:, 3] -= 0.2  # Lower NIR
water_spectra = np.clip(water_spectra, 0, 1)

for i in range(n_bands):
    extended_data["signals"][f"band_{i}"].extend(water_spectra[:, i].tolist())

extended_data["metadata"]["sample_id"].extend([f"sample_{i}" for i in range(50, 75)])
extended_data["target"]["label"].extend(["water"] * 25)
extended_data["target"]["value"].extend([2] * 25)
extended_data["target"]["encoding"]["water"] = 2

extended_dataset = TabularDatasetData.from_dict(extended_data)

print("Displaying multi-class spectral signals...")
display_signals(
    extended_dataset,
    figsize=(12, 8),
    colormap="viridis",
    x_label="Spectral Bands",
    y_label="Reflectance Values",
)
