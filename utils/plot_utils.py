import collections
import logging
from itertools import product
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spectral as sp
from funcy import log_durations
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector

from utils.utils import get_logger

logger = get_logger(name="plot_utils")


def pixels_select_click(image):
    image_display = image.to_display()
    coordinates_list = np.empty([0,3])
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image_display)
    fig.tight_layout()

    def onclick(event):
        nonlocal coordinates_list, fig
        if plt.get_current_fig_manager().toolbar.mode != '': return
        logger.info(f'Pressed coordinate: X = {event.xdata}, Y = {event.ydata}')
        x_coor = event.xdata
        y_coor = event.ydata
        coordinates_list = np.vstack((coordinates_list, [x_coor, y_coor, 1]))

        ax.scatter(
            int(x_coor),
            int(y_coor),
            marker='x',
            c="red",
        )
        fig.canvas.draw()

    def accept(event):
        if event.key == "enter":
            plt.close()

    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect("key_press_event", accept)
    plt.show(block=True)

    return pd.DataFrame(coordinates_list.astype("int"), columns=["x","y","z"])


def pixels_select_lasso(image):
    image_display = image.to_display()
    x, y = np.meshgrid(
        np.arange(image_display.shape[1]), np.arange(image_display.shape[0]))
    pix = np.vstack((x.flatten(), y.flatten())).T

    fig, ax = plt.subplots(1, 1)
    ax.imshow(image_display)
    fig.tight_layout()

    indices = 0
    selected_areas = []

    def onselect(verts, eps=1e-8):
        nonlocal indices
        p = Path(verts)
        indices = p.contains_points(pix, radius=1)

    def onrelease(_):
        nonlocal indices, selected_areas
        coordinates_list = np.hstack((pix[indices], np.ones((pix[indices].shape[0], 1))))
        coordinates_df = pd.DataFrame(coordinates_list.astype("int"), columns=["x","y","z"])
        selected_areas.append(coordinates_df.drop_duplicates())

    def accept(event):
        if event.key == "enter":
            plt.close()

    lasso = LassoSelector(ax, onselect)
    cid = fig.canvas.mpl_connect('button_release_event', onrelease)
    fig.canvas.mpl_connect("key_press_event", accept)

    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()
    plt.show(block=True)

    return selected_areas


@log_durations(logging.info)
def display_images(images, images_selected_areas=None, colors="red"):
    # TODO make some additional arguments checking points
    if isinstance(images, SimpleNamespace):
        images = [images.cam1, images.cam2]
        images_selected_areas = [images_selected_areas.cam1, images_selected_areas.cam2]

    num_images = len(images)
    fig, axes = plt.subplots(1, num_images)
    # in case only images from one camera provided
    if not isinstance(axes, (collections.Sequence, np.ndarray)):
        axes = [axes]

    for image_idx, image in enumerate(images):
        image_display = image.to_display()
        axes[image_idx].imshow(image_display)
        axes[image_idx].axis("off")

    fig.tight_layout()
    mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()

    for image_idx, selected_areas in enumerate(images_selected_areas):
        if selected_areas:
            for area_idx, selected_area in enumerate(selected_areas):
                if not isinstance(colors, list):
                    color = colors
                else:
                    color = colors[image_idx][area_idx]
                axes[image_idx].scatter(
                    selected_area.x,
                    selected_area.y,
                    lw=0,
                    marker='o',
                    c=color,
                    s=(72.0 / fig.dpi) ** 2,
                )

    def accept(event):
        if event.key == "enter":
            plt.close()
    fig.canvas.mpl_connect("key_press_event", accept)

