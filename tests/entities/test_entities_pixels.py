import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from siapy.entities import Pixels

iterable = [(1, 2), (3, 4), (5, 6)]
iterable_homo = [(1, 2, 1), (3, 4, 1), (5, 6, 1)]


def test_from_iterable():
    pixels = Pixels.from_iterable(iterable)
    assert isinstance(pixels, Pixels)
    assert pixels.df.equals(
        pd.DataFrame(iterable, columns=[Pixels.coords.U, Pixels.coords.V])
    )


def test_df():
    df = pd.DataFrame(iterable, columns=["u", "v"])
    pixels = Pixels(df)
    assert pixels.df.equals(df)


def test_df_homogenious():
    df = pd.DataFrame(iterable, columns=[Pixels.coords.U, Pixels.coords.V])
    pixels = Pixels(df)
    df_homogenious = pixels.df_homogenious()
    assert df_homogenious.equals(
        pd.DataFrame(
            iterable_homo, columns=[Pixels.coords.U, Pixels.coords.V, Pixels.coords.H]
        )
    )


def test_u():
    df = pd.DataFrame(iterable, columns=[Pixels.coords.U, Pixels.coords.V])
    pixels = Pixels(df)
    expected_x = pd.Series([1, 3, 5], name=Pixels.coords.U)
    assert pixels.u().equals(expected_x)


def test_v():
    df = pd.DataFrame(iterable, columns=[Pixels.coords.U, Pixels.coords.V])
    pixels = Pixels(df)
    expected_y = pd.Series([2, 4, 6], name=Pixels.coords.V)
    assert pixels.v().equals(expected_y)


def test_to_numpy():
    df = pd.DataFrame(iterable, columns=[Pixels.coords.U, Pixels.coords.V])
    pixels = Pixels(df)
    expected_array = df.to_numpy()
    assert np.array_equal(pixels.to_numpy(), expected_array)


def test_save_and_load_to_parquet():
    pixels = Pixels.from_iterable(iterable)
    with TemporaryDirectory() as tmpdir:
        parquet_file = Path(tmpdir, "test_pixels.parquet")
        pixels.save_to_parquet(parquet_file)
        assert os.path.exists(parquet_file)
        loaded_pixels = Pixels.load_from_parquet(parquet_file)
        assert isinstance(loaded_pixels, Pixels)
        assert loaded_pixels.df.equals(pixels.df)
