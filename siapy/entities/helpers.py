import itertools

from shapely.geometry import MultiPoint, Point
from shapely.prepared import prep as shapely_prep

from siapy.core.exceptions import InvalidTypeError
from siapy.entities import Shape, Signatures, SpectralImage


def get_signatures_within_convex_hull(image: SpectralImage, shape: Shape) -> list[Signatures]:
    image_xarr = image.to_xarray()
    signatures = []

    if shape.is_point:
        for g in shape.geometry:
            if isinstance(g, MultiPoint):
                points = list(g.geoms)
            elif isinstance(g, Point):
                points = [g]
            else:
                raise InvalidTypeError(
                    input_value=g,
                    allowed_types=(Point, MultiPoint),
                    message="Geometry must be Point or MultiPoint",
                )
            signals = []
            pixels = []
            for p in points:
                signals.append(image_xarr.sel(x=p.x, y=p.y, method="nearest").values)
                pixels.append((p.x, p.y))

            signatures.append(Signatures.from_signals_and_pixels(signals, pixels))

    else:
        for hull in shape.convex_hull:
            minx, miny, maxx, maxy = hull.bounds

            x_coords = image_xarr.x[(image_xarr.x >= minx) & (image_xarr.x <= maxx)].values
            y_coords = image_xarr.y[(image_xarr.y >= miny) & (image_xarr.y <= maxy)].values

            if len(x_coords) == 0 or len(y_coords) == 0:
                continue

            # Create a prepared geometry for faster contains check
            prepared_hull = shapely_prep(hull)

            signals = []
            pixels = []
            for x, y in itertools.product(x_coords, y_coords):
                point = Point(x, y)
                # Check if point is: inside the hull or intersects with the hull
                if prepared_hull.contains(point) or prepared_hull.intersects(point):
                    try:
                        signal = image_xarr.sel(x=x, y=y).values
                    except (KeyError, IndexError):
                        continue
                    signals.append(signal)
                    pixels.append((x, y))

            signatures.append(Signatures.from_signals_and_pixels(signals, pixels))

    return signatures
