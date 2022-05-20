from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spectral as sp
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector

from utils import utils

logger = utils.get_logger(name="plot_utils")


def pixels_select_click(image):
    image_display = image.to_display()
    coordinates_list = np.empty([0,3])
    fig = plt.figure()
    def onclick(event):
        nonlocal coordinates_list, fig
        if plt.get_current_fig_manager().toolbar.mode != '': return
        logger.info(f'Pressed coordinate: X = {event.xdata}, Y = {event.ydata}')
        x_coor = event.xdata
        y_coor = event.ydata
        coordinates_list = np.vstack((coordinates_list, [x_coor, y_coor, 1]))

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.imshow(image_display)
    plt.show()

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
        selected_areas.append(coordinates_df)

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


def display_images(images, images_selected_areas, colors):
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images)

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
                axes[image_idx].scatter(
                    selected_area.x,
                    selected_area.y,
                    lw=0,
                    c=colors[area_idx],
                    s=(72.0 / fig.dpi) ** 2,
                )



