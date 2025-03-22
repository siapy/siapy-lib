from siapy.entities.images.rasterio_lib import RasterioLibImage


def test_open_valid(configs):
    raster = RasterioLibImage.open(configs.image_micasense_merged)
    assert isinstance(raster, RasterioLibImage)
