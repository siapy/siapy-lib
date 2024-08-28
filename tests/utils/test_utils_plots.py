import pytest

from siapy.datasets.schemas import (
    RegressionTarget,
    TabularDatasetData,
)
from siapy.utils.enums import InteractiveButtonsEnum
from siapy.utils.plots import (
    display_image_with_areas,
    display_multiple_images_with_areas,
    display_signals,
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
    selected_areas = pixels_select_lasso(image_vnir, selector_props={"color": "blue"})
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


@pytest.mark.manual
def test_display_signals():
    data = {
        "pixels": {
            "0": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "1": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        },
        "signals": {
            "0": [1, 2, 3, 1, 1, 5, 7, 6, 7, 8],
            "1": [3, 4, 5, 6, 4, 10, 11, 12, 11, 12],
        },
        "metadata": {
            "0": [
                "meta1",
                "meta1",
                "meta1",
                "meta1",
                "meta1",
                "meta1",
                "meta1",
                "meta1",
                "meta1",
                "meta1",
            ],
            "1": [
                "meta2",
                "meta2",
                "meta2",
                "meta2",
                "meta2",
                "meta2",
                "meta2",
                "meta2",
                "meta2",
                "meta2",
            ],
        },
        "target": {
            "label": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"],
            "value": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "encoding": ["x", "y"],
        },
    }
    dataset = TabularDatasetData.from_dict(data)
    display_signals(dataset)
    dataset.target = RegressionTarget.from_iterable(
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    )
    with pytest.raises(ValueError):
        display_signals(dataset)
