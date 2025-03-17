import numpy as np
import pandas as pd
import pytest

from siapy.core.exceptions import InvalidInputError
from siapy.features.spectral_indices import (
    _convert_str_to_list,
    compute_spectral_indices,
    get_spectral_indices,
)


def test_convert_str_to_list():
    input_data = "R"
    expected_output = ["R"]
    assert _convert_str_to_list(input_data) == expected_output
    input_data = ["R", "G", "B"]
    expected_output = ["R", "G", "B"]
    assert _convert_str_to_list(input_data) == expected_output


def test_get_spectral_indices_valid():
    bands_acronym = ["R", "G", "B"]
    spectral_indices = get_spectral_indices(bands_acronym)
    for _, meta in spectral_indices.items():
        assert set(meta.bands).issubset(set(bands_acronym))


def test_get_spectral_indices_invalid():
    bands_acronym = ["not-correct"]
    with pytest.raises(InvalidInputError):
        get_spectral_indices(bands_acronym)


def test_compute_spectral_indices():
    columns = ["R", "G"]
    spectral_indices = get_spectral_indices(columns)
    data = pd.DataFrame(np.random.default_rng(seed=0).random((5, 2)), columns=columns)
    compute_spectral_indices(data, spectral_indices.keys())
    columns[1] = "not-correct"
    data = pd.DataFrame(np.random.default_rng(seed=0).random((5, 2)), columns=columns)
    with pytest.raises(InvalidInputError):
        compute_spectral_indices(
            data,
            spectral_indices.keys(),
        )


def test_compute_spectral_indices_with_map():
    data = pd.DataFrame(
        np.random.default_rng(seed=0).random((5, 2)),
        columns=["R", "not-correct"],
    )
    spectral_indices = get_spectral_indices(["R", "G"])
    with pytest.raises(InvalidInputError):
        compute_spectral_indices(
            data,
            spectral_indices.keys(),
            {"not-correct": "not-correct2"},
        )
    compute_spectral_indices(data, spectral_indices.keys(), {"not-correct": "G"})
