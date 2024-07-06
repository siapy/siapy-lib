import pytest

from siapy.utils.plots import (
    display_selected_areas,
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
    display_selected_areas(image_vnir, selected_areas, color="blue")
