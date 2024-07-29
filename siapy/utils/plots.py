import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib.widgets import Button, LassoSelector

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


def display_image_with_selected_areas(
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


def display_multiple_images_with_selected_areas(
    images_with_areas: list[tuple[ImageType, Pixels | list[Pixels]]],
    *,
    color: str = "red",
):
    num_images = len(images_with_areas)
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 5, 5))

    if num_images == 1:
        axes = [axes]

    for ax, (image, selected_areas) in zip(axes, images_with_areas):
        if not isinstance(selected_areas, list):
            selected_areas = [selected_areas]

        image_display = validate_image_to_numpy_3channels(image)
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


def interactive_buttons():
    from .enums import InteractiveButtonsEnum

    flag = InteractiveButtonsEnum.REPEAT

    def repeat(event):
        nonlocal flag
        logger.info("Pressed repeat button.")
        plt.close()
        flag = InteractiveButtonsEnum.REPEAT

    def save(event):
        nonlocal flag
        logger.info("Pressed save button.")
        plt.close()
        flag = InteractiveButtonsEnum.SAVE

    def skip(event):
        nonlocal flag
        logger.info("Pressed skip button.")
        plt.close()
        flag = InteractiveButtonsEnum.SKIP

    axcolor = "lightgoldenrodyellow"
    position = plt.axes([0.9, 0.1, 0.1, 0.04])
    button_save = Button(position, "Save", color=axcolor, hovercolor="0.975")
    button_save.on_clicked(save)
    position = plt.axes([0.9, 0.15, 0.1, 0.04])
    button_repeat = Button(position, "Repeat", color=axcolor, hovercolor="0.975")
    button_repeat.on_clicked(repeat)
    position = plt.axes([0.9, 0.2, 0.1, 0.04])
    button_skip = Button(position, "Skip", color=axcolor, hovercolor="0.975")
    button_skip.on_clicked(skip)
    plt.show()
