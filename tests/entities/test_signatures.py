import numpy as np
import pandas as pd

from siapy.entities import Pixels, Signatures

#     image = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
#     pixels = Pixels.from_iterable([(0, 0), (1, 1)])
#     signatures = Signatures.from_array_and_pixels(image, pixels)
#     assert isinstance(signatures, Signatures)
#     expected_df = pd.DataFrame(
#         [(0, 0, [1, 2, 3]), (1, 1, [10, 11, 12])],
#         columns=[Pixels.U, Pixels.V, Signatures.SIG],
#     )
#     pd.testing.assert_frame_equal(signatures.df, expected_df)


# def test_df():
#     df = pd.DataFrame(
#         [(0, 0, [1, 2, 3]), (1, 1, [10, 11, 12])],
#         columns=[Pixels.U, Pixels.V, Signatures.SIG],
#     )
#     signatures = Signatures(df)
#     pd.testing.assert_frame_equal(signatures.df, df)


def test_from_array_and_pixels():
    image = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    pixels_df = pd.DataFrame({"u": [0, 1], "v": [0, 1]})
    pixels = Pixels(pixels_df)

    signatures = Signatures.from_array_and_pixels(image, pixels)

    expected_data = pd.DataFrame(
        {
            (Signatures.PIX, "u"): [0, 1],
            (Signatures.PIX, "v"): [0, 1],
            (Signatures.SIG, 0): [1, 10],
            (Signatures.SIG, 1): [2, 11],
            (Signatures.SIG, 2): [3, 12],
        }
    )
    pd.testing.assert_frame_equal(signatures.df, expected_data)


# def test_signals():
#     data = pd.DataFrame(
#         {
#             (Signatures.PIX, "u"): [0, 1],
#             (Signatures.PIX, "v"): [0, 1],
#             (Signatures.SIG, 0): [1, 10],
#             (Signatures.SIG, 1): [2, 11],
#             (Signatures.SIG, 2): [3, 12],
#         }
#     )
#     signatures = Signatures._create(data)

#     expected_data = pd.DataFrame({0: [1, 10], 1: [2, 11], 2: [3, 12]})
#     pd.testing.assert_frame_equal(signatures.signals().df, expected_data)


def test_pixels():
    data = pd.DataFrame(
        {
            (Signatures.PIX, "u"): [0, 1],
            (Signatures.PIX, "v"): [0, 1],
            (Signatures.SIG, 0): [1, 10],
            (Signatures.SIG, 1): [2, 11],
            (Signatures.SIG, 2): [3, 12],
        }
    )
    signatures = Signatures._create(data)

    expected_data = pd.DataFrame({"u": [0, 1], "v": [0, 1]})
    pd.testing.assert_frame_equal(signatures.pixels().df, expected_data)
