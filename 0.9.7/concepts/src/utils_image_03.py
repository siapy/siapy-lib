from siapy.entities import SpectralImage
from siapy.entities.shapes import Shape
from siapy.utils.images import (
    calculate_correction_factor_from_panel,
    convert_radiance_image_to_reflectance,
    spy_save_image,
)

# Load radiance image
radiance_img = SpectralImage.spy_open(header_path="radiance_data.hdr")

# Method 1: Using labeled geometric shape for panel area
panel_shape = Shape.from_rectangle(x_min=200, y_min=350, x_max=300, y_max=400, label="reference_panel")
radiance_img.geometric_shapes.append(panel_shape)

correction_factor = calculate_correction_factor_from_panel(
    image=radiance_img,
    panel_reference_reflectance=0.2,  # 20% reflectance panel
    panel_shape_label="reference_panel",
)

# Method 2: Using entire image (when image contains only panel)
# panel_img = SpectralImage.spy_open(header_path="panel_only.hdr")
# correction_factor = calculate_correction_factor_from_panel(
#     image=panel_img,
#     panel_reference_reflectance=0.2
# )

# Convert radiance to reflectance
reflectance_img = convert_radiance_image_to_reflectance(image=radiance_img, panel_correction=correction_factor)

# Save reflectance image with enhanced metadata
metadata = radiance_img.metadata.copy()
metadata.update({"data_type": "reflectance", "reference_panel": "20% Spectralon", "processing_date": "2025-06-19"})
spy_save_image(image=reflectance_img, save_path="output/reflectance.hdr", metadata=metadata)
