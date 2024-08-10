import warnings
from typing import Any, Iterable

import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message="pkg_resources is deprecated as an API",
    )
    import spyndex  # type: ignore


def _convert_str_to_list(bands_acronym: Any) -> Any:
    if isinstance(bands_acronym, str):
        bands_acronym = [bands_acronym]
    return bands_acronym


def get_spectral_indices(
    bands_acronym: str | Iterable[str],
) -> dict[str, spyndex.axioms.SpectralIndex]:
    bands_acronym = _convert_str_to_list(bands_acronym)
    bands_acronym_set = set(bands_acronym)

    if not bands_acronym_set.issubset(list(spyndex.bands)):
        raise ValueError(
            f"Invalid input argument for 'bands_acronym'. \n"
            f"Received: {bands_acronym_set}. \n"
            f"Possible options are: {list(spyndex.bands)}. \n"
            "Please ensure that all elements in 'bands_acronym' are valid band acronyms."
        )

    spectral_indexes = {}
    for name in spyndex.indices.to_dict():
        index = spyndex.indices[name]
        if set(index.bands).issubset(bands_acronym_set):
            spectral_indexes[name] = index

    return spectral_indexes


def compute_spectral_indices(
    data: pd.DataFrame,
    spectral_indices: str | Iterable[str],
    bands_map: dict[str, str] | None = None,
    remove_nan_and_constants: bool = True,
) -> pd.DataFrame:
    spectral_indices = _convert_str_to_list(spectral_indices)

    params = {}
    for band in data.columns:
        if bands_map is not None and band in bands_map.keys():
            if bands_map[band] not in list(spyndex.bands):
                raise ValueError(
                    f"Invalid band mapping: '{bands_map[band]}' is not a recognized band acronym. \n"
                    f"Received mapping: {band} -> {bands_map[band]}. \n"
                    f"Possible options are: {list(spyndex.bands)}. \n"
                    "Please ensure that all values in 'bands_map' are valid band acronyms."
                )
            params[bands_map[band]] = data[band]
        else:
            if band not in list(spyndex.bands):
                raise ValueError(
                    f"Invalid band: '{band}' is not a recognized band acronym. \n"
                    f"Possible options are: {list(spyndex.bands)}. \n"
                    "Please ensure that all columns in 'data' are valid band acronyms."
                )
            params[band] = data[band]

    df = spyndex.computeIndex(index=list(spectral_indices), params=params)
    if remove_nan_and_constants:
        # Drop columns with inf or NaN values
        df = df.drop(df.columns[df.isin([np.inf, -np.inf, np.nan]).any()], axis=1)
        # Drop columns with constant values
        df = df.drop(df.columns[df.nunique() == 1], axis=1)
    return df
