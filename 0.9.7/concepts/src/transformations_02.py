import numpy as np

from siapy.entities import SpectralImage
from siapy.transformations.image import add_gaussian_noise

# Create a sample spectral image
rng = np.random.default_rng(seed=42)
image_array = rng.random((100, 100, 10))
image = SpectralImage.from_numpy(image_array)

# Add Gaussian noise to the image
noisy_image = add_gaussian_noise(
    image,
    mean=0.0,  # Center the noise around zero
    std=0.1,  # Standard deviation of the noise
    clip_to_max=True,  # Prevent values from exceeding the original range
)

image_np = image.to_numpy()

print(f"Original image range: [{image_np.min():.3f}, {image_np.max():.3f}]")
print(f"Noisy image range: [{noisy_image.min():.3f}, {noisy_image.max():.3f}]")

# Compare signal-to-noise ratio
signal_power = np.mean(image_np**2)
noise_power = np.mean((noisy_image - image) ** 2)
snr_db = 10 * np.log10(signal_power / noise_power)
print(f"Signal-to-Noise Ratio: {snr_db:.2f} dB")
