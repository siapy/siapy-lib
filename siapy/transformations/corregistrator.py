from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from siapy.entities.pixels import CoordinateInput, Pixels, validate_pixel_input

__all__ = [
    "map_affine_approx_2d",
    "affine_matx_2d",
    "align",
    "transform",
]


def map_affine_approx_2d(
    points_ref: NDArray[np.floating[Any]], points_mov: NDArray[np.floating[Any]]
) -> NDArray[np.floating[Any]]:
    """Affine transformation"""
    # U = T*X -> T = U*X'(X*X')^-1
    matx_2d = points_ref.transpose() @ points_mov @ np.linalg.inv(points_mov.transpose() @ points_mov)
    return matx_2d


def affine_matx_2d(
    scale: tuple[float, float] | Sequence[float] = (1, 1),
    trans: tuple[float, float] | Sequence[float] = (0, 0),
    rot: float = 0,
    shear: tuple[float, float] | Sequence[float] = (0, 0),
) -> NDArray[np.floating[Any]]:
    """Create arbitrary affine transformation matrix"""
    rot = rot * np.pi / 180
    matx_scale = np.array(((scale[0], 0, 0), (0, scale[1], 0), (0, 0, 1)))
    matx_trans = np.array(((1, 0, trans[0]), (0, 1, trans[1]), (0, 0, 1)))
    matx_rot = np.array(
        (
            (np.cos(rot), -np.sin(rot), 0),
            (np.sin(rot), np.cos(rot), 0),
            (0, 0, 1),
        )
    )
    matx_shear = np.array(((1, shear[0], 0), (shear[1], 1, 0), (0, 0, 1)))
    matx_2d = np.dot(matx_trans, np.dot(matx_shear, np.dot(matx_rot, matx_scale)))
    return matx_2d


def align(
    pixels_ref: Pixels | pd.DataFrame | Iterable[CoordinateInput],
    pixels_mov: Pixels | pd.DataFrame | Iterable[CoordinateInput],
    *,
    eps: float = 1e-6,
    max_iter: int = 50,
    plot_progress: bool = False,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Align interactive corresponding points"""
    pixels_ref = validate_pixel_input(pixels_ref)
    pixels_mov = validate_pixel_input(pixels_mov)

    points_ref = pixels_ref.df_homogenious().to_numpy()
    points_mov = pixels_mov.df_homogenious().to_numpy()

    matrices = []
    errors = []
    idx = 0
    if plot_progress:
        points_mov_orig = points_mov
        fig = plt.figure()
        ax = fig.add_subplot(111)

    while True:
        points_ref_corr = np.array(points_ref)
        points_mov_corr = np.array(points_mov)

        matx_2d_combined = map_affine_approx_2d(points_ref_corr, points_mov_corr)
        points_mov = np.dot(points_mov, matx_2d_combined.transpose())

        matrices.append(matx_2d_combined)
        errors.append(np.sqrt(np.sum((points_ref_corr[:, :2] - points_mov_corr[:, :2]) ** 2)))
        idx = idx + 1

        # check for convergence
        matx_diff = np.abs(matx_2d_combined - affine_matx_2d())
        if idx > max_iter or np.all(matx_diff < eps):
            break

    matx_2d_combined = affine_matx_2d()  # initialize with identity matrix
    for matx_2d in matrices:
        if plot_progress:
            points_mov_corr = np.dot(points_mov_orig, matx_2d_combined.transpose())
            ax.clear()
            ax.plot(points_ref[:, 0], points_ref[:, 1], "ob")
            ax.plot(points_mov_corr[:, 0], points_mov_corr[:, 1], "om")
            fig.canvas.draw()
            plt.pause(1)

        # multiply all matrices to get the final transformation
        matx_2d_combined = np.dot(matx_2d, matx_2d_combined)

    errors_np = np.array(errors)

    return matx_2d_combined, errors_np


def transform(
    pixels: Pixels | pd.DataFrame | Iterable[CoordinateInput], transformation_matx: NDArray[np.floating[Any]]
) -> Pixels:
    """Transform pixels"""
    pixels = validate_pixel_input(pixels)
    points_transformed = np.dot(pixels.df_homogenious().to_numpy(), transformation_matx.transpose())
    points_transformed = np.round(points_transformed[:, :2]).astype("int")
    return Pixels.from_iterable(points_transformed)
