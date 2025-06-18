from siapy.features.spectral_indices import get_spectral_indices

# Get indices computable from Red and Green bands
bands = ["R", "G"]
available_indices = get_spectral_indices(bands)
print(f"Found {len(available_indices)} indices")

# Display the names and long names of the available indices
for name, index in list(available_indices.items()):
    print(f"{name}: {index.long_name}")
