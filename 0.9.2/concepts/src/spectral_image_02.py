from siapy.entities import SpectralImage
from siapy.entities.images import RasterioLibImage

# Load from GeoTIFF or other raster formats (uses rioxarray/rasterio python library)
image_rio = RasterioLibImage.open(
    filepath="path/to/image.tif",
)
image = SpectralImage(image_rio)

# Or you can use the class method to load the image directly
image = SpectralImage.rasterio_open(filepath="path/to/image.tif")
