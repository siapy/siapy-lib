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


def test_x():
    df = pd.DataFrame(iterable, columns=[Pixels.coords.U, Pixels.coords.V])
    pixels = Pixels(df)
    expected_x = pd.Series([1, 3, 5], name=Pixels.coords.U)
    assert pixels.x().equals(expected_x)


def test_y():
    df = pd.DataFrame(iterable, columns=[Pixels.coords.U, Pixels.coords.V])
    pixels = Pixels(df)
    expected_y = pd.Series([2, 4, 6], name=Pixels.coords.V)
    assert pixels.y().equals(expected_y)
