from siapy.entities import SpectralImage
from siapy.utils.images import blockfy_image, calculate_image_background_percentage, spy_merge_images_by_specter

# Merge VNIR and SWIR images
vnir_image = SpectralImage.spy_open(header_path="vnir_data.hdr")
swir_image = SpectralImage.spy_open(header_path="swir_data.hdr")

merged_image = spy_merge_images_by_specter(
    image_original=vnir_image,
    image_to_merge=swir_image,
    save_path="output/merged_vnir_swir.hdr",
    auto_metadata_extraction=True,
)

# Image blocking for large dataset processing
large_image = SpectralImage.spy_open(header_path="large_image.hdr")

blocks = blockfy_image(
    image=large_image,
    p=50,  # block height
    q=50,  # block width
)

# Background analysis
bg_percentage = calculate_image_background_percentage(large_image)
print(f"Background pixels: {bg_percentage:.2f}%")
