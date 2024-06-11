import numpy as np
import pandas as pd
import pytest

from siapy.entities import Pixels, SigFilterEnum, Signatures


def test_from_array_and_pixels():
    image = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    pixels = Pixels.from_iterable([(0, 0), (1, 1)])
    signatures = Signatures.from_array_and_pixels(image, pixels)
    assert isinstance(signatures, Signatures)
    expected_df = pd.DataFrame(
        [(0, 0, [1, 2, 3]), (1, 1, [10, 11, 12])],
        columns=[Pixels.U, Pixels.V, Signatures.SIG],
    )
    pd.testing.assert_frame_equal(signatures.df, expected_df)


def test_df():
    df = pd.DataFrame(
        [(0, 0, [1, 2, 3]), (1, 1, [10, 11, 12])],
        columns=[Pixels.U, Pixels.V, Signatures.SIG],
    )
    signatures = Signatures(df)
    pd.testing.assert_frame_equal(signatures.df, df)


def test_df_filtered():
    df = pd.DataFrame(
        [(0, 0, [1, 2, 3]), (1, 1, [10, 11, 12])],
        columns=[Pixels.U, Pixels.V, Signatures.SIG],
    )
    signatures = Signatures(df)

    # Test filtering for signatures
    expected_df = df[[Signatures.SIG]]
    pd.testing.assert_frame_equal(
        signatures.df_filtered(SigFilterEnum.SIGNATURES), expected_df
    )

    # Test filtering for pixels
    expected_df = df[[Pixels.U, Pixels.V]]
    pd.testing.assert_frame_equal(
        signatures.df_filtered(SigFilterEnum.PIXELS), expected_df
    )

    # Test no filtering
    pd.testing.assert_frame_equal(signatures.df_filtered(), df)

    # Test invalid argument
    with pytest.raises(ValueError):
        signatures.df_filtered("invalid")


# def test_to_numpy():
#     df = pd.DataFrame(
#         [(0, 0, [1, 2, 3]), (1, 1, [10, 11, 12])],
#         columns=[Pixels.U, Pixels.V, Signatures.SIG],
#     )
#     signatures = Signatures(df)

#     # Test conversion to numpy array for signatures
#     expected_array = np.vstack(df[[Signatures.SIG]].to_numpy())
#     np.testing.assert_array_equal(
#         signatures.to_numpy(SigFilterEnum.SIGNATURES), expected_array
#     )

#     # Test conversion to numpy array for pixels
#     expected_array = np.vstack(df[[Pixels.U, Pixels.V]].to_numpy())
#     np.testing.assert_array_equal(
#         signatures.to_numpy(SigFilterEnum.PIXELS), expected_array
#     )

#     # Test conversion to numpy array with no filtering
#     expected_array = np.vstack(df.to_numpy())
#     np.testing.assert_array_equal(signatures.to_numpy(), expected_array)

#     # Test both valid ways to convert dataframe to numpy array
#     np.testing.assert_array_equal(signatures.df.to_numpy(), signatures.to_numpy())
