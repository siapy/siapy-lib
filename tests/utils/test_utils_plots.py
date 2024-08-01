import pytest

from siapy.utils.enums import InteractiveButtonsEnum
from siapy.utils.plots import (
    display_image_with_areas,
    display_multiple_images_with_areas,
    pixels_select_click,
    pixels_select_lasso,
)


@pytest.mark.manual
def test_pixels_select_click_manual(spectral_images):
    image_vnir = spectral_images.vnir
    pixels_select_click(image_vnir.to_display())


@pytest.mark.manual
def test_pixels_select_lasso_manual(spectral_images):
    image_vnir = spectral_images.vnir
    selected_areas = pixels_select_lasso(image_vnir)
    display_image_with_areas(image_vnir, selected_areas, color="blue")


@pytest.mark.manual
def test_display_multiple_images_with_areas(spectral_images, corresponding_pixels):
    image_vnir = spectral_images.vnir
    image_swir = spectral_images.swir
    selected_areas_vnir = corresponding_pixels.vnir
    selected_areas_swir = corresponding_pixels.swir
    # selected_areas_vnir = pixels_select_lasso(image_vnir)
    # selected_areas_swir = pixels_select_lasso(image_swir)
    out = display_multiple_images_with_areas(
        images_with_areas=[
            (image_vnir, selected_areas_vnir),
            (image_swir, selected_areas_swir),
        ],
        color="blue",
    )
    assert isinstance(out, InteractiveButtonsEnum)
