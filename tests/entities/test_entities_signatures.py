import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from siapy.core.exceptions import InvalidInputError
from siapy.entities import Pixels, Signatures
from siapy.entities.signatures import Signals

## Signals


def test_signals_len():
    signals_df = pd.DataFrame([[1, 2, 4], [3, 4, 3]])
    signals = Signals(signals_df)
    assert len(signals) == len(signals_df)


def test_signals_getitem():
    signals_df = pd.DataFrame([[1, 2, 4], [3, 4, 3]])
    signals = Signals(signals_df)

    # Single index access
    single_signal = signals[0]
    assert isinstance(single_signal, Signals)
    assert len(single_signal) == 1
    assert single_signal.df.equals(pd.DataFrame([[1, 2, 4]], columns=signals_df.columns))

    # Slice access
    sliced_signals = signals[0:1]
    assert isinstance(sliced_signals, Signals)
    assert len(sliced_signals) == 1
    assert sliced_signals.df.equals(pd.DataFrame([[1, 2, 4]], columns=signals_df.columns))

    # List access
    list_signals = signals[[0]]
    assert isinstance(list_signals, Signals)
    assert len(list_signals) == 1
    assert list_signals.df.equals(pd.DataFrame([[1, 2, 4]], columns=signals_df.columns))


def test_signals_from_iterable():
    data_list = [[1, 2, 3], [4, 5, 6]]
    signals = Signals.from_iterable(data_list)
    assert isinstance(signals, Signals)
    assert len(signals) == 2
    assert signals.df.equals(pd.DataFrame(data_list))


def test_signals_to_numpy():
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


## Signatures


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


def test_signatures_len():
    pixels_df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]})
    signals_df = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
    pixels = Pixels(pixels_df)
    signals = Signals(signals_df)
    signatures = Signatures(pixels, signals)

    assert len(signatures) == 3


def test_signatures_getitem():
    pixels_df = pd.DataFrame({"x": [0, 1, 2, 3], "y": [0, 1, 2, 3]})
    signals_df = pd.DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]])
    pixels = Pixels(pixels_df)
    signals = Signals(signals_df)
    signatures = Signatures(pixels, signals)

    # Single index access
    single_sig = signatures[1]
    assert isinstance(single_sig, Signatures)
    assert len(single_sig) == 1
    assert single_sig.pixels.df.equals(pd.DataFrame({"x": [1], "y": [1]}, index=[1]))
    assert single_sig.signals.df.equals(pd.DataFrame([[3, 4]], index=[1]))

    # Slice access
    sliced_sig = signatures[1:3]
    assert isinstance(sliced_sig, Signatures)
    assert len(sliced_sig) == 2
    assert sliced_sig.pixels.df.equals(pd.DataFrame({"x": [1, 2], "y": [1, 2]}, index=[1, 2]))
    assert sliced_sig.signals.df.equals(pd.DataFrame([[3, 4], [5, 6]], index=[1, 2]))

    # List access
    list_sig = signatures[[0, 3]]
    assert isinstance(list_sig, Signatures)
    assert len(list_sig) == 2
    assert list_sig.pixels.df.equals(pd.DataFrame({"x": [0, 3], "y": [0, 3]}, index=[0, 3]))
    assert list_sig.signals.df.equals(pd.DataFrame([[1, 2], [7, 8]], index=[0, 3]))


def test_validate_inputs():
    # Test successful validation with matching row counts
    pixels_df = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
    signals_df = pd.DataFrame([[1, 2], [3, 4]])
    pixels = Pixels(pixels_df)
    signals = Signals(signals_df)

    # Should work without raising exception
    signatures = Signatures(pixels, signals)
    assert len(signatures) == 2

    # Test validation failure with mismatched row counts
    pixels_df_long = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]})
    signals_df_short = pd.DataFrame([[1, 2], [3, 4]])
    pixels_long = Pixels(pixels_df_long)
    signals_short = Signals(signals_df_short)

    # Should raise InvalidInputError
    with pytest.raises(InvalidInputError, match="Pixels and signals must have the same number of rows"):
        Signatures(pixels_long, signals_short)

    # Test the reverse case
    pixels_df_short = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
    signals_df_long = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
    pixels_short = Pixels(pixels_df_short)
    signals_long = Signals(signals_df_long)

    # Should also raise InvalidInputError
    with pytest.raises(InvalidInputError, match="Pixels and signals must have the same number of rows"):
        Signatures(pixels_short, signals_long)


def test_signatures_from_dict():
    data = {"pixels": {"x": [0, 1, 2], "y": [3, 4, 5]}, "signals": {"0": [10, 20, 30], "1": [40, 50, 60]}}

    signatures = Signatures.from_dict(data)

    assert isinstance(signatures, Signatures)
    assert len(signatures) == 3
    assert signatures.pixels.df.equals(pd.DataFrame({"x": [0, 1, 2], "y": [3, 4, 5]}))
    assert signatures.signals.df.equals(pd.DataFrame({"0": [10, 20, 30], "1": [40, 50, 60]}))


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


def test_signatures_to_dict():
    pixels_df = pd.DataFrame({"x": [0, 1], "y": [2, 3]})
    signals_df = pd.DataFrame({"A": [10, 20], "B": [30, 40]})
    pixels = Pixels(pixels_df)
    signals = Signals(signals_df)
    signatures = Signatures(pixels, signals)

    result = signatures.to_dict()

    assert isinstance(result, dict)
    assert "pixels" in result
    assert "signals" in result

    # Convert back to DataFrames for comparison
    pixels_from_dict = pd.DataFrame(result["pixels"])
    signals_from_dict = pd.DataFrame(result["signals"])

    assert pixels_from_dict.equals(pixels_df)
    assert signals_from_dict.equals(signals_df)


def test_signatures_reset_index():
    # Create DataFrame with non-standard indices
    pixels_df = pd.DataFrame({"x": [0, 1], "y": [2, 3]}, index=[5, 10])
    signals_df = pd.DataFrame({"A": [10, 20], "B": [30, 40]}, index=[5, 10])
    pixels = Pixels(pixels_df)
    signals = Signals(signals_df)
    signatures = Signatures(pixels, signals)

    reset_signatures = signatures.reset_index()

    assert len(reset_signatures) == len(signatures)
    assert reset_signatures.pixels.df.index.equals(pd.RangeIndex(start=0, stop=2))
    assert reset_signatures.signals.df.index.equals(pd.RangeIndex(start=0, stop=2))

    # Data values should remain the same
    assert np.array_equal(reset_signatures.pixels.df["x"].values, signatures.pixels.df["x"].values)
    assert np.array_equal(reset_signatures.pixels.df["y"].values, signatures.pixels.df["y"].values)
    assert np.array_equal(reset_signatures.signals.df["A"].values, signatures.signals.df["A"].values)
    assert np.array_equal(reset_signatures.signals.df["B"].values, signatures.signals.df["B"].values)


def test_signatures_copy():
    # Create original Signatures object
    pixels_df = pd.DataFrame({"x": [0, 1, 2], "y": [3, 4, 5]})
    signals_df = pd.DataFrame({"A": [10, 20, 30], "B": [40, 50, 60]})
    pixels = Pixels(pixels_df)
    signals = Signals(signals_df)
    original = Signatures(pixels, signals)

    # Create a copy
    copied = original.copy()

    # Verify the copy is a new object but with equal data
    assert copied is not original
    assert copied.pixels is not original.pixels
    assert copied.signals is not original.signals
    assert copied.pixels.df is not original.pixels.df
    assert copied.signals.df is not original.signals.df

    # Verify the data is the same
    pd.testing.assert_frame_equal(copied.pixels.df, original.pixels.df)
    pd.testing.assert_frame_equal(copied.signals.df, original.signals.df)

    # Modify the copy and verify the original is unchanged
    copied.pixels.df.loc[0, "x"] = 999
    copied.signals.df.loc[0, "A"] = 888

    assert original.pixels.df.loc[0, "x"] == 0  # Original should be unchanged
    assert original.signals.df.loc[0, "A"] == 10  # Original should be unchanged
    assert copied.pixels.df.loc[0, "x"] == 999  # Copy should be changed
    assert copied.signals.df.loc[0, "A"] == 888  # Copy should be changed
