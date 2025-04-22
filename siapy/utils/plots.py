import sys
from enum import Enum, auto
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.widgets import Button, LassoSelector
from numpy.typing import NDArray

from siapy.core.exceptions import InvalidInputError
from siapy.core.logger import logger
from siapy.core.types import ImageType
from siapy.datasets.schemas import ClassificationTarget, TabularDatasetData
from siapy.entities import Pixels
from siapy.utils.image_validators import validate_image_to_numpy_3channels

__all__ = [
    "pixels_select_click",
    "pixels_select_lasso",
    "display_image_with_areas",
    "display_multiple_images_with_areas",
    "display_signals",
    "InteractiveButtonsEnum",
]


def pixels_select_click(image: ImageType) -> Pixels:
    image_display = validate_image_to_numpy_3channels(image)

    coordinates = []
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image_display)
    fig.tight_layout()
    enter_clicked = 0

    def onclick(event: Any) -> None:
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

    def accept(event: Any) -> None:
        nonlocal enter_clicked
        if event.key == "enter":
            logger.info("Enter clicked.")
            enter_clicked = 1
            plt.close()

    def onexit(event: Any) -> None:
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


def pixels_select_lasso(image: ImageType, selector_props: dict[str, Any] | None = None) -> list[Pixels]:
    image_display = validate_image_to_numpy_3channels(image)

    x, y = np.meshgrid(np.arange(image_display.shape[1]), np.arange(image_display.shape[0]))
    pixes_all_stack = np.vstack((x.flatten(), y.flatten())).T

    fig, ax = plt.subplots(1, 1)
    ax.imshow(image_display)
    fig.tight_layout()

    indices: NDArray[np.bool_] = np.array([], dtype=bool)
    indices_list: list[NDArray[np.bool_]] = []
    enter_clicked = 0

    def onselect(vertices_selected: list[tuple[float, float]]) -> None:
        logger.info("Selected.")
        nonlocal indices
        path = Path(vertices_selected)
        indices = path.contains_points(pixes_all_stack)

    def onrelease(event: Any) -> None:
        nonlocal indices, indices_list
        indices_list.append(indices)

    def onexit(event: Any) -> None:
        nonlocal enter_clicked
        if not enter_clicked:
            logger.info("Exiting application.")
            plt.close()
            sys.exit(0)

    def accept(event: Any) -> None:
        nonlocal enter_clicked
        if event.key == "enter":
            logger.info("Enter clicked.")
            enter_clicked = 1
            plt.close()

    props = selector_props if selector_props is not None else {"color": "red", "linewidth": 2, "linestyle": "-"}
    lasso = LassoSelector(ax, onselect, props=props)  # noqa: F841 type: ignore[reportUnusedVariable]
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


def display_image_with_areas(
    image: ImageType,
    areas: Pixels | list[Pixels],
    *,
    color: str = "red",
) -> None:
    if not isinstance(areas, list):
        areas = [areas]

    image_display = validate_image_to_numpy_3channels(image)
    fig, ax = plt.subplots()
    ax.imshow(image_display)

    for pixels in areas:
        ax.scatter(
            pixels.x(),
            pixels.y(),
            lw=0,
            marker="o",
            c=color,
            s=(72.0 / fig.dpi) ** 2,
        )

    plt.show()


class InteractiveButtonsEnum(Enum):
    SAVE = auto()
    REPEAT = auto()
    SKIP = auto()


def display_multiple_images_with_areas(
    images_with_areas: list[tuple[ImageType, Pixels | list[Pixels]]],
    *,
    color: str = "red",
    plot_interactive_buttons: bool = True,
) -> InteractiveButtonsEnum | None:
    num_images = len(images_with_areas)
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 5, 5))

    if isinstance(axes, Axes):
        axes = np.array([axes])

    for ax, (image, selected_areas) in zip(axes, images_with_areas):
        if not isinstance(selected_areas, list):
            selected_areas = [selected_areas]

        image_display = validate_image_to_numpy_3channels(image)
        ax.imshow(image_display)

        for pixels in selected_areas:
            ax.scatter(
                pixels.x(),
                pixels.y(),
                lw=0,
                marker="o",
                c=color,
                s=(72.0 / fig.dpi) ** 2,
            )
    if plot_interactive_buttons:
        return interactive_buttons()

    plt.show()
    return None


def interactive_buttons() -> InteractiveButtonsEnum:
    flag = InteractiveButtonsEnum.REPEAT

    def repeat(event: Any) -> None:
        nonlocal flag
        logger.info("Pressed repeat button.")
        plt.close()
        flag = InteractiveButtonsEnum.REPEAT

    def save(event: Any) -> None:
        nonlocal flag
        logger.info("Pressed save button.")
        plt.close()
        flag = InteractiveButtonsEnum.SAVE

    def skip(event: Any) -> None:
        nonlocal flag
        logger.info("Pressed skip button.")
        plt.close()
        flag = InteractiveButtonsEnum.SKIP

    axcolor = "lightgoldenrodyellow"
    position = plt.axes((0.9, 0.1, 0.1, 0.04))
    button_save = Button(position, "Save", color=axcolor, hovercolor="0.975")
    button_save.on_clicked(save)
    position = plt.axes((0.9, 0.15, 0.1, 0.04))
    button_repeat = Button(position, "Repeat", color=axcolor, hovercolor="0.975")
    button_repeat.on_clicked(repeat)
    position = plt.axes((0.9, 0.2, 0.1, 0.04))
    button_skip = Button(position, "Skip", color=axcolor, hovercolor="0.975")
    button_skip.on_clicked(skip)
    plt.show()
    return flag


def display_signals(
    data: TabularDatasetData,
    *,
    figsize: tuple[int, int] = (6, 4),
    dpi: int = 150,
    colormap: str = "viridis",
    x_label: str = "Spectral bands",
    y_label: str = "",
    label_fontsize: int = 14,
    tick_params_label_size: int = 12,
    legend_fontsize: int = 10,
    legend_frameon: bool = True,
) -> None:
    if not isinstance(data.target, ClassificationTarget):
        raise InvalidInputError(
            input_value=data.target,
            message="The target must be an instance of ClassificationTarget.",
        )

    signals = data.signatures.signals.df.copy()
    target = data.target.model_copy()
    y_data_encoded = target.value
    classes = list(target.encoding.to_dict().values())

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cmap = plt.get_cmap(colormap)
    unique_labels = np.unique(y_data_encoded)
    no_colors = len(unique_labels)

    if no_colors > 2:
        colors = list(cmap(np.linspace(0, 1, no_colors)))
    else:
        colors = ["darkgoldenrod", "forestgreen"]

    x_values = list(range(len(signals.columns)))

    grouped_data = signals.groupby(y_data_encoded.to_numpy())
    mean_values = grouped_data.mean()
    std_values = grouped_data.std()

    for idx in unique_labels:
        mean = mean_values.loc[idx].tolist()
        std = std_values.loc[idx].tolist()
        ax.plot(x_values, mean, color=colors[idx], label=classes[idx], alpha=0.6)
        ax.fill_between(
            x_values,
            [m - s for m, s in zip(mean, std)],
            [m + s for m, s in zip(mean, std)],
            color=colors[idx],
            alpha=0.2,
        )

    custom_lines = []
    for idx in unique_labels:
        custom_lines.append(Line2D([0], [0], color=colors[idx], lw=2))

    ax.set_ylabel(y_label, fontsize=label_fontsize)
    ax.set_xlabel(x_label, fontsize=label_fontsize)
    ax.tick_params(axis="both", which="major", labelsize=tick_params_label_size)
    ax.tick_params(axis="both", which="minor", labelsize=tick_params_label_size)
    # ax.set_ylim([0, 1])
    # ax.spines["bottom"].set_linewidth(2)
    # ax.spines["left"].set_linewidth(2)
    ax.spines["right"].set_linewidth(0)
    ax.spines["top"].set_linewidth(0)
    ax.set_xticks(x_values)
    ax.set_xticklabels(signals.columns, rotation=0)
    ax.legend(
        loc="upper left",
        fontsize=legend_fontsize,
        framealpha=1,
        frameon=legend_frameon,
    )
    plt.show()
