from siapy.entities import Shape, SpectralImage
from siapy.entities.helpers import get_signatures_within_convex_hull


def test_properties(configs):
    raster = SpectralImage.rasterio_open(configs.image_micasense_merged)
    point_shape = Shape.open_shapefile(configs.shapefile_point)
    buffer_shape = Shape.open_shapefile(configs.shapefile_buffer)
    raster.geometric_shapes.shapes = [point_shape, buffer_shape]
    get_signatures_within_convex_hull(raster, point_shape)
    get_signatures_within_convex_hull(raster, buffer_shape)
