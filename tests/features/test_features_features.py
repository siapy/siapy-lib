import pandas as pd
from sklearn.datasets import make_classification, make_regression

from siapy.features import (
    AutoSpectralIndicesClassification,
    AutoSpectralIndicesRegression,
)
from siapy.features.helpers import FeatureSelectorConfig
from siapy.features.spectral_indices import (
    compute_spectral_indices,
    get_spectral_indices,
)


def test_auto_spectral_indices_classification():
    columns = ["R", "G"]
    spectral_indices = get_spectral_indices(columns)
    X, y = make_classification(n_samples=100, n_features=2, n_classes=2, random_state=0, n_redundant=0)
    data = pd.DataFrame(X, columns=columns)
    target = pd.Series(y)
    df_direct = compute_spectral_indices(data, spectral_indices.keys())

    config = FeatureSelectorConfig(k_features=5)
    auto_clf = AutoSpectralIndicesClassification(spectral_indices, selector_config=config, merge_with_original=False)
    df_selected = auto_clf.fit_transform(data, target)
    pd.testing.assert_frame_equal(df_selected, df_direct[df_selected.columns])


def test_auto_spectral_indices_regression():
    columns = ["R", "G"]
    spectral_indices = get_spectral_indices(columns)
    X, y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=0)
    data = pd.DataFrame(X, columns=columns)
    target = pd.Series(y)
    df_direct = compute_spectral_indices(data, spectral_indices.keys())

    config = FeatureSelectorConfig(k_features=5)
    auto_reg = AutoSpectralIndicesRegression(spectral_indices, selector_config=config, merge_with_original=False)
    df_selected = auto_reg.fit_transform(data, target)
    pd.testing.assert_frame_equal(df_selected, df_direct[df_selected.columns])
