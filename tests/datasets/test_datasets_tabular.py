from siapy.datasets import TabularDataset
from siapy.entities import Pixels, Shape, SpectralImageSet


def test_tabular(spectral_images):
    pixels_input = [(10, 15), (60, 66)]
    pixels = Pixels.from_iterable(pixels_input)
    rectangle = Shape.from_shape_type(shape_type="rectangle", pixels=pixels)

    spectral_images.vnir.geometric_shapes.append(rectangle)
    spectral_images.swir.geometric_shapes.append(rectangle)

    images = [
        spectral_images.vnir,
        spectral_images.swir,
        spectral_images.vnir,
    ]

    image_set = SpectralImageSet(images)
    dataset = TabularDataset(image_set)
    dataset.generate()
