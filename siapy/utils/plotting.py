import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
from PIL.Image import Image

from siapy.core.logger import logger
from siapy.entities import Pixels, SpectralImage

ImageType = SpectralImage | np.ndarray | Image


def validate_and_convert_image(image: ImageType) -> np.ndarray:
    if isinstance(image, SpectralImage):
        image_display = np.array(image.to_display())
    elif isinstance(image, Image):
        image_display = np.array(image)
    elif (
        isinstance(image, np.ndarray) and len(image.shape) == 3 and image.shape[-1] == 3
    ):
        image_display = image
    else:
        raise ValueError("Image must be convertable to 3d numpy array.")
    return image_display


def pixels_select_click(image: ImageType) -> Pixels:
    image_display = validate_and_convert_image(image)

    coordinates = []
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image_display)
    fig.tight_layout()
    enter_clicked = 0

    def onclick(event):
        nonlocal coordinates, fig
        logger.info(f"Pressed coordinate: X = {event.xdata}, Y = {event.ydata}")
        x_coor = round(event.xdata)
        y_coor = round(event.ydata)
        coordinates.append([x_coor, y_coor])

        ax.scatter(
            int(x_coor),
            int(y_coor),
            marker="x",
            c="red",
        )
        fig.canvas.draw()

    def accept(event):
        nonlocal enter_clicked
        if event.key == "enter":
            logger.info("Enter clicked.")
            enter_clicked = 1
            plt.close()

    def onexit(event):
        nonlocal enter_clicked
        if not enter_clicked:
            logger.info("Exiting application.")
            plt.close()
            sys.exit(0)

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", accept)
    fig.canvas.mpl_connect("close_event", onexit)
    plt.show()
    return Pixels.from_iterable(coordinates)


def pixels_select_lasso(image: ImageType) -> list[Pixels]:
    image_display = validate_and_convert_image(image)

    x, y = np.meshgrid(
        np.arange(image_display.shape[1]), np.arange(image_display.shape[0])
    )
    pixes_all_stack = np.vstack((x.flatten(), y.flatten())).T

    fig, ax = plt.subplots(1, 1)
    ax.imshow(image_display)
    fig.tight_layout()

    indices = 0
    indices_list = []
    enter_clicked = 0

    def onselect(vertices_selected, eps=1e-8):
        logger.info("Selected.")
        nonlocal indices
        path = Path(vertices_selected)
        indices = np.nonzero(path.contains_points(pixes_all_stack))[0]

    def onrelease(_):
        nonlocal indices, indices_list
        indices_list.append(indices)

    def onexit(event):
        nonlocal enter_clicked
        if not enter_clicked:
            logger.info("Exiting application.")
            plt.close()
            sys.exit(0)

    def accept(event):
        nonlocal enter_clicked
        if event.key == "enter":
            logger.info("Enter clicked.")
            enter_clicked = 1
            plt.close()

    lasso = LassoSelector(ax, onselect)
    fig.canvas.mpl_connect("button_release_event", onrelease)
    fig.canvas.mpl_connect("close_event", onexit)
    fig.canvas.mpl_connect("key_press_event", accept)

    plt.show()

    selected_areas = []
    for indices in indices_list:
        coordinates = pixes_all_stack[indices]
        selected_areas.append(Pixels.from_iterable(coordinates))

    logger.info(f"Number of selected areas: {len(selected_areas)}")
    return selected_areas


def display_selected_areas(
    image: ImageType,
    selected_areas: list[Pixels],
    *,
    color: str = "red",
):
    image_display = validate_and_convert_image(image)

    fig, ax = plt.subplots()
    ax.imshow(image_display)

    for pixels in selected_areas:
        polygon = Polygon(pixels.to_numpy(), color=color)
        ax.add_patch(polygon)

    plt.show()
