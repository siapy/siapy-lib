import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from siapy.core.exceptions import InvalidInputError
from siapy.entities import Pixels, Signatures
from siapy.entities.signatures import Signals


def test_signals():
    df = pd.DataFrame({"A": [1, 2, 3]})
    signals = Signals(df)

    assert signals.df.equals(df)
    assert np.array_equal(signals.to_numpy(), df.to_numpy())


def test_signals_mean():
    signals_df = pd.DataFrame([[1, 2, 4, 6], [3, 4, 3, 5]])
    signals = Signals(signals_df)
    signals_mean = signals.mean()
    assert np.array_equal(signals_mean, [2.0, 3.0, 3.5, 5.5])
    assert signals_mean.shape == (4,)


def test_signals_save_and_load_to_parquet():
    signals_df = pd.DataFrame([[1, 2, 4, 6], [3, 4, 3, 5]])
    signals = Signals(signals_df)
    with TemporaryDirectory() as tmpdir:
        parquet_file = Path(tmpdir, "test_signals.parquet")
        signals.save_to_parquet(parquet_file)
        assert os.path.exists(parquet_file)
        loaded_signals = Signals.load_from_parquet(parquet_file)
        assert isinstance(loaded_signals, Signals)
        assert loaded_signals.df.equals(signals.df)


def test_signatures_create():
    pixels_df = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
    signals_df = pd.DataFrame([[1, 2], [3, 4]])
    pixels = Pixels(pixels_df)
    signals = Signals(signals_df)
    signatures = Signatures(pixels, signals)

    assert signatures.pixels == pixels
    assert signatures.signals == signals
    assert signatures.to_dataframe().equals(pd.concat([pixels_df, signals_df], axis=1))
    assert np.array_equal(
        signatures.to_numpy(),
        pd.concat([pixels_df, signals_df], axis=1).to_numpy(),
    )


def test_signatures_from_array_and_pixels():
    pixels_df = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
    pixels = Pixels(pixels_df)
    image = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    from_array_and_pixels = Signatures.from_array_and_pixels(image, pixels)

    assert from_array_and_pixels.pixels == pixels
    assert from_array_and_pixels.signals.df.equals(pd.DataFrame(list(image[pixels.v(), pixels.u(), :])))


def test_signatures_from_array_and_pixels_invalid_dimensions():
    pixels_df = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
    pixels = Pixels(pixels_df)
    image_2d = np.array([[1, 2], [3, 4]])
    with pytest.raises(InvalidInputError, match="Expected a 3-dimensional array, but got 2-dimensional array"):
        Signatures.from_array_and_pixels(image_2d, pixels)
    image_4d = np.array([[[[1]]]])
    with pytest.raises(InvalidInputError, match="Expected a 3-dimensional array, but got 4-dimensional array"):
        Signatures.from_array_and_pixels(image_4d, pixels)


def test_signatures_from_array_and_pixels_coordinate_bounds():
    pixels_df = pd.DataFrame({"x": [0, 5], "y": [0, 5]})
    pixels = Pixels(pixels_df)
    image = np.zeros((2, 2, 2))
    with pytest.raises(InvalidInputError, match="Pixel coordinates exceed image dimensions"):
        Signatures.from_array_and_pixels(image, pixels)


def test_signatures_from_dataframe():
    df = pd.DataFrame({"x": [0, 1], "y": [0, 1], "0": [1, 2], "1": [3, 4]})
    signatures = Signatures.from_dataframe(df)
    expected_pixels_df = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
    expected_signals_df = pd.DataFrame({"0": [1, 2], "1": [3, 4]})

    assert signatures.pixels.df.equals(expected_pixels_df)
    assert signatures.signals.df.equals(expected_signals_df)

    df_missing_u = pd.DataFrame({"V": [0, 1], "0": [1, 2], "1": [3, 4]})

    with pytest.raises(InvalidInputError):
        Signatures.from_dataframe(df_missing_u)

    df_missing_v = pd.DataFrame({"U": [0, 1], "0": [1, 2], "1": [3, 4]})

    with pytest.raises(InvalidInputError):
        Signatures.from_dataframe(df_missing_v)


def test_signatures_save_and_load_to_parquet():
    df = pd.DataFrame({"x": [0, 1], "y": [0, 1], "0": [1, 2], "1": [3, 4]})
    signatures = Signatures.from_dataframe(df)
    with TemporaryDirectory() as tmpdir:
        parquet_file = Path(tmpdir, "test_signatures.parquet")
        signatures.save_to_parquet(parquet_file)
        assert os.path.exists(parquet_file)
        loaded_signatures = Signatures.open_parquet(parquet_file)
        assert isinstance(loaded_signatures, Signatures)
        assert loaded_signatures.to_dataframe().equals(signatures.to_dataframe())


def test_signatures_dataframe_multiindex_conversion():
    pixels_df = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
    signals_df = pd.DataFrame({"ch1": [1, 2], "ch2": [3, 4]})
    pixels = Pixels(pixels_df)
    signals = Signals(signals_df)
    signatures = Signatures(pixels, signals)

    df_multi = signatures.to_dataframe_multiindex()

    assert isinstance(df_multi.columns, pd.MultiIndex)
    assert "pixel" in df_multi.columns.get_level_values(0)
    assert "signal" in df_multi.columns.get_level_values(0)

    # Convert back
    new_signatures = Signatures.from_dataframe_multiindex(df_multi)
    assert new_signatures.pixels == signatures.pixels
    assert new_signatures.signals.df.equals(signatures.signals.df)


def test_signatures_from_dataframe_multiindex_invalid_input():
    regular_df = pd.DataFrame({"x": [0], "y": [0]})
    with pytest.raises(InvalidInputError):
        Signatures.from_dataframe_multiindex(regular_df)
