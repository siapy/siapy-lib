import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector

from siapy.core.logger import logger
from siapy.core.types import ImageType
from siapy.entities import Pixels
from siapy.utils.validators import validate_image_to_numpy_3channels


def pixels_select_click(image: ImageType) -> Pixels:
    image_display = validate_image_to_numpy_3channels(image)

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
    image_display = validate_image_to_numpy_3channels(image)

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
        indices = path.contains_points(pixes_all_stack)

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

    lasso = LassoSelector(ax, onselect)  # noqa: F841
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
    selected_areas: Pixels | list[Pixels],
    *,
    color: str = "red",
):
    if not isinstance(selected_areas, list):
        selected_areas = [selected_areas]

    image_display = validate_image_to_numpy_3channels(image)
    fig, ax = plt.subplots()
    ax.imshow(image_display)

    for pixels in selected_areas:
        ax.scatter(
            pixels.u(),
            pixels.v(),
            lw=0,
            marker="o",
            c=color,
            s=(72.0 / fig.dpi) ** 2,
        )

    plt.show()
