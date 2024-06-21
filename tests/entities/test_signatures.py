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
    assert signatures.df().equals(pd.concat([pixels_df, signals_df], axis=1))
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
