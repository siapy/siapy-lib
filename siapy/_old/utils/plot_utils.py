import collections
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inpoly import inpoly2
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, LassoSelector
from sklearn import preprocessing

from siapy.utils.utils import get_logger

logger = get_logger(name="plot_utils")


def pixels_select_click(image):
    image_display = image.to_display()
    coordinates_list = np.empty([0, 3])
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image_display)
    fig.tight_layout()
    enter_clicked = 0

    def onclick(event):
        nonlocal coordinates_list, fig
        if plt.get_current_fig_manager().toolbar.mode != "":
            return
        logger.info(f"Pressed coordinate: X = {event.xdata}, Y = {event.ydata}")
        x_coor = event.xdata
        y_coor = event.ydata
        coordinates_list = np.vstack((coordinates_list, [x_coor, y_coor, 1]))

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

    return pd.DataFrame(coordinates_list.astype("int"), columns=["x", "y", "z"])


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


# @log_durations(logging.info)
def display_images(images, images_selected_areas=None, colors="red"):
    # TODO make some additional arguments checking points
    if isinstance(images, SimpleNamespace):
        images_ = [images.cam1]
        if images_selected_areas is not None:
            images_selected_areas_ = [images_selected_areas.cam1]
        if images.cam2 is not None:
            images_.append(images.cam2)
            if images_selected_areas is not None:
                images_selected_areas_.append(images_selected_areas.cam2)
        images = images_
        if images_selected_areas is not None:
            images_selected_areas = images_selected_areas_

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

    if images_selected_areas is not None:
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
                        marker="o",
                        c=color,
                        s=(72.0 / fig.dpi) ** 2,
                    )

    def accept(event):
        if event.key == "enter":
            logger.info("Enter clicked. Exiting.")
            plt.close()

    fig.canvas.mpl_connect("key_press_event", accept)


def segmentation_buttons():
    flag = "repeat"

    def repeat(event):
        nonlocal flag
        logger.info("Pressed repeat button.")
        plt.close()
        flag = "repeat"

    def save(event):
        nonlocal flag
        logger.info("Pressed save button.")
        plt.close()
        flag = "save"

    def skip(event):
        nonlocal flag
        logger.info("Pressed skip button.")
        plt.close()
        flag = "skip"

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

    return flag


def plot_signatures(signatures, groups_labels, *, x_scat=None):
    """Plot signatures with corresponding wavelenght or consequtive wavelength number

    Args:
        signatures (list): list of lists/signatures
        labels (list): list of labels
    """
    signatures = np.array(signatures)
    labels = np.array(groups_labels)
    # if all labels equal to nan than all are left for plotting
    if len(np.unique(labels)) == 1 and str(labels[0]) == "nan":
        pass
    else:
        # remove nans in case more unique labels in group_labels
        # nans are removed in this case
        indices = [idx for idx, lab in enumerate(labels) if str(lab) != "nan"]
        signatures = signatures[indices]
        labels = labels[indices]

    transformer = preprocessing.LabelEncoder()
    transformer.fit(labels)
    labels = transformer.transform(labels)

    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(15)

    ax = fig.add_subplot(1, 1, 1)
    # ax.set_title(title, fontsize=14)
    # ax.set_ylabel(y_label, fontsize=24)
    # ax.set_xlabel(x_label, fontsize=24)
    ax.tick_params(axis="both", which="major", labelsize=22)
    ax.tick_params(axis="both", which="minor", labelsize=22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cmap = plt.get_cmap("viridis")
    no_colors = len(np.unique(labels))
    colors = cmap(np.linspace(0, 1, no_colors))

    if x_scat is None:
        x_scat = list(np.arange(len(signatures[0])))

    for obj in range(len(signatures)):
        ax.plot(x_scat, signatures[obj], color=colors[labels[obj]], alpha=0.6)

    custom_lines = []
    for idx in range(no_colors):
        custom_lines.append(Line2D([0], [0], color=colors[idx], lw=2))

    labels = transformer.inverse_transform(list(range(no_colors)))
    ax.legend(custom_lines, [str(num) for num in labels], fontsize=22)
    plt.show()
