import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from siapy.entities import Pixels, Signatures
from siapy.entities.signatures import Signals, SignaturesFilter


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


def test_signatures_filter_create():
    pixels_df = pd.DataFrame({"u": [0, 1], "v": [0, 1]})
    signals_df = pd.DataFrame([[1, 2], [3, 4]])
    pixels = Pixels(pixels_df)
    signals = Signals(signals_df)
    signatures = Signatures._create(pixels, signals)

    assert signatures.filter() == SignaturesFilter(pixels, signals)


def test_signatures_filter_build():
    pixels_df = pd.DataFrame({"u": [0, 1], "v": [0, 1]})
    signals_df = pd.DataFrame([[1, 2], [3, 4]])
    pixels = Pixels(pixels_df)
    signals = Signals(signals_df)
    filter = SignaturesFilter(pixels, signals)

    assert filter.build() == Signatures._create(pixels, signals)


def test_signatures_filter_with_slice():
    pixels_df = pd.DataFrame({"u": [0, 1], "v": [0, 1]})
    signals_df = pd.DataFrame([[1, 2], [3, 4]])
    pixels = Pixels(pixels_df)
    signals = Signals(signals_df)
    filter = SignaturesFilter(pixels, signals)

    rows_filter = filter.rows(slice(0, 1))
    assert rows_filter.pixels.df.equals(pixels_df.iloc[slice(0, 1)])
    assert rows_filter.signals.df.equals(signals_df.iloc[slice(0, 1)])

    cols_filter = filter.cols(slice(0, 1))
    assert cols_filter.pixels.df.equals(pixels_df)
    assert cols_filter.signals.df.equals(signals_df.iloc[:, slice(0, 1)])


def test_signatures_filter_with_list_int():
    pixels_df = pd.DataFrame({"u": [0, 1], "v": [0, 1]})
    signals_df = pd.DataFrame([[1, 2], [3, 4]])
    pixels = Pixels(pixels_df)
    signals = Signals(signals_df)
    filter = SignaturesFilter(pixels, signals)

    rows_filter = filter.rows([0])
    assert rows_filter.pixels.df.equals(pixels_df.iloc[[0]])
    assert rows_filter.signals.df.equals(signals_df.iloc[[0]])

    cols_filter = filter.cols([0])
    assert cols_filter.pixels.df.equals(pixels_df)
    assert cols_filter.signals.df.equals(signals_df.iloc[:, [0]])


def test_signatures_filter_with_list_bool():
    pixels_df = pd.DataFrame({"u": [0, 1], "v": [0, 1]})
    signals_df = pd.DataFrame([[1, 2], [3, 4]])
    pixels = Pixels(pixels_df)
    signals = Signals(signals_df)
    filter = SignaturesFilter(pixels, signals)

    rows_filter = filter.rows([True, False])
    assert rows_filter.pixels.df.equals(pixels_df.iloc[[True, False]])
    assert rows_filter.signals.df.equals(signals_df.iloc[[True, False]])

    cols_filter = filter.cols([True, False])
    assert cols_filter.pixels.df.equals(pixels_df)
    assert cols_filter.signals.df.equals(signals_df.iloc[:, [True, False]])


def test_signatures_create():
    pixels_df = pd.DataFrame({"u": [0, 1], "v": [0, 1]})
    signals_df = pd.DataFrame([[1, 2], [3, 4]])
    pixels = Pixels(pixels_df)
    signals = Signals(signals_df)
    signatures = Signatures._create(pixels, signals)

    assert signatures.pixels == pixels
    assert signatures.signals == signals
    assert signatures.to_dataframe().equals(pd.concat([pixels_df, signals_df], axis=1))
    assert np.array_equal(
        signatures.to_numpy(), pd.concat([pixels_df, signals_df], axis=1).to_numpy()
    )


def test_signatures_raise_error():
    with pytest.raises(RuntimeError):
        Signatures()


def test_signatures_from_array_and_pixels():
    pixels_df = pd.DataFrame({"u": [0, 1], "v": [0, 1]})
    pixels = Pixels(pixels_df)
    image = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    from_array_and_pixels = Signatures.from_array_and_pixels(image, pixels)

    assert from_array_and_pixels.pixels == pixels
    assert from_array_and_pixels.signals.df.equals(
        pd.DataFrame(list(image[pixels.v(), pixels.u(), :]))
    )


def test_signatures_from_dataframe():
    df = pd.DataFrame({"u": [0, 1], "v": [0, 1], "0": [1, 2], "1": [3, 4]})
    signatures = Signatures.from_dataframe(df)
    expected_pixels_df = pd.DataFrame({"u": [0, 1], "v": [0, 1]})
    expected_signals_df = pd.DataFrame({"0": [1, 2], "1": [3, 4]})

    assert signatures.pixels.df.equals(expected_pixels_df)
    assert signatures.signals.df.equals(expected_signals_df)

    df_missing_u = pd.DataFrame({"V": [0, 1], "0": [1, 2], "1": [3, 4]})

    with pytest.raises(ValueError):
        Signatures.from_dataframe(df_missing_u)

    df_missing_v = pd.DataFrame({"U": [0, 1], "0": [1, 2], "1": [3, 4]})

    with pytest.raises(ValueError):
        Signatures.from_dataframe(df_missing_v)


def test_signatures_save_and_load_to_parquet():
    df = pd.DataFrame({"u": [0, 1], "v": [0, 1], "0": [1, 2], "1": [3, 4]})
    signatures = Signatures.from_dataframe(df)
    with TemporaryDirectory() as tmpdir:
        parquet_file = Path(tmpdir, "test_signatures.parquet")
        signatures.save_to_parquet(parquet_file)
        assert os.path.exists(parquet_file)
        loaded_signatures = Signatures.load_from_parquet(parquet_file)
        assert isinstance(loaded_signatures, Signatures)
        assert loaded_signatures.to_dataframe().equals(signatures.to_dataframe())
