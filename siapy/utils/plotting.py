import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from inpoly import inpoly2
from matplotlib.widgets import LassoSelector
from PIL.Image import Image

from siapy.core.logger import logger
from siapy.entities import Pixels, SpectralImage


def validate_and_convert_image(image: SpectralImage | np.ndarray | Image) -> np.ndarray:
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


def pixels_select_click(image: SpectralImage | np.ndarray | Image):
    image_display = validate_and_convert_image(image)

    coordinates_list = []
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image_display)
    fig.tight_layout()
    enter_clicked = 0

    def onclick(event):
        nonlocal coordinates_list, fig
        # if plt.get_current_fig_manager().toolbar.mode != "":
        #     return
        logger.info(f"Pressed coordinate: X = {event.xdata}, Y = {event.ydata}")
        x_coor = round(event.xdata)
        y_coor = round(event.ydata)
        coordinates_list.append([x_coor, y_coor])

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
    plt.show(block=True)
    return Pixels.from_iterable(coordinates_list)


def pixels_select_lasso(image):
    image_display = image.to_display()
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
        indices, _ = inpoly2(pixes_all_stack, vertices_selected)

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

    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()
    plt.show(block=True)

    selected_areas = []
    for indices in indices_list:
        coordinates_list = np.hstack(
            (pixes_all_stack[indices], np.ones((pixes_all_stack[indices].shape[0], 1)))
        )
        coordinates_df = pd.DataFrame(
            coordinates_list.astype("int"), columns=["x", "y", "z"]
        )
        selected_areas.append(coordinates_df.drop_duplicates())
    logger.info(f"Number of selected areas: {len(selected_areas)}")

    return selected_areas
