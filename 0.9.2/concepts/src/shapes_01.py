from siapy.entities import Pixels, Shape

# Create a point
point = Shape.from_point(10, 20)

# Create a polygon from pixels
pixels = Pixels.from_iterable([(0, 0), (10, 0), (10, 10), (0, 10)])
polygon = Shape.from_polygon(pixels)

# Load from shapefile
shape = Shape.open_shapefile("path/to/shapefile.shp")
