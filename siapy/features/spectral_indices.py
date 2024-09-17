import warnings
from typing import Any, Iterable

import numpy as np
import pandas as pd

from siapy.core.exceptions import InvalidInputError

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message="pkg_resources is deprecated as an API",
    )
    import spyndex  # type: ignore

__all__ = [
    "get_spectral_indices",
    "compute_spectral_indices",
]


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
        raise InvalidInputError(
            {
                "received_bands_acronym": bands_acronym_set,
                "valid_bands_acronym": list(spyndex.bands),
            },
            "Invalid input argument for 'bands_acronym'. Please ensure that all elements in 'bands_acronym' are valid band acronyms.",
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
                raise InvalidInputError(
                    {
                        "received_band_mapping": bands_map[band],
                        "valid_bands_acronym": list(spyndex.bands),
                    },
                    f"Invalid band mapping is not a recognized band acronym. \n"
                    f"Received mapping: {band} -> {bands_map[band]}. \n"
                    "Please ensure that all values in 'bands_map' are valid band acronyms.",
                )
            params[bands_map[band]] = data[band]
        else:
            if band not in list(spyndex.bands):
                raise InvalidInputError(
                    {
                        "received_band": band,
                        "valid_bands_acronym": list(spyndex.bands),
                    },
                    f"Invalid band: '{band}' is not a recognized band acronym. \n"
                    "Please ensure that all columns in 'data' are valid band acronyms.",
                )
            params[band] = data[band]

    df = spyndex.computeIndex(index=list(spectral_indices), params=params)
    if remove_nan_and_constants:
        # Drop columns with inf or NaN values
        df = df.drop(df.columns[df.isin([np.inf, -np.inf, np.nan]).any()], axis=1)
        # Drop columns with constant values
        df = df.drop(df.columns[df.nunique() == 1], axis=1)
    return df
