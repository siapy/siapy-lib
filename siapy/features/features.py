from typing import Iterable, Literal

import numpy as np
import pandas as pd
from autofeat import AutoFeatClassifier, AutoFeatRegressor  # type: ignore
from sklearn.base import BaseEstimator, TransformerMixin

from siapy.features.helpers import FeatureSelectorConfig, feature_selector_factory
from siapy.features.spectral_indices import compute_spectral_indices
from siapy.utils.general import set_random_seed


class AutoFeatClassification(AutoFeatClassifier):
    def __init__(
        self,
        *,
        categorical_cols: list | None = None,
        feateng_cols: list | None = None,
        units: dict | None = None,
        feateng_steps: int = 2,
        featsel_runs: int = 5,
        max_gb: int | None = None,
        transformations: list | tuple = ("1/", "exp", "log", "abs", "sqrt", "^2", "^3"),
        apply_pi_theorem: bool = True,
        always_return_numpy: bool = False,
        n_jobs: int = 1,
        verbose: int = 0,
        random_seed: int | None = None,
    ):
        self.random_seed = random_seed
        set_random_seed(self.random_seed)
        super().__init__(
            categorical_cols=categorical_cols,
            feateng_cols=feateng_cols,
            units=units,
            feateng_steps=feateng_steps,
            featsel_runs=featsel_runs,
            max_gb=max_gb,
            transformations=transformations,
            apply_pi_theorem=apply_pi_theorem,
            always_return_numpy=always_return_numpy,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    def fit(self, data: np.ndarray | pd.DataFrame, target: np.ndarray | pd.DataFrame):
        set_random_seed(self.random_seed)
        super().fit(data, target)

    def transform(self, data: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
        set_random_seed(self.random_seed)
        data_transformed = super().transform(data)
        return data_transformed

    def fit_transform(
        self, data: np.ndarray | pd.DataFrame, target: np.ndarray | pd.DataFrame
    ) -> np.ndarray | pd.DataFrame:
        set_random_seed(self.random_seed)
        data_transformed = super().fit_transform(data, target)
        return data_transformed


class AutoFeatRegression(AutoFeatRegressor):
    def __init__(
        self,
        *,
        categorical_cols: list | None = None,
        feateng_cols: list | None = None,
        units: dict | None = None,
        feateng_steps: int = 2,
        featsel_runs: int = 5,
        max_gb: int | None = None,
        transformations: list | tuple = ("1/", "exp", "log", "abs", "sqrt", "^2", "^3"),
        apply_pi_theorem: bool = True,
        always_return_numpy: bool = False,
        n_jobs: int = 1,
        verbose: int = 0,
        random_seed: int | None = None,
    ):
        self.random_seed = random_seed
        set_random_seed(self.random_seed)
        super().__init__(
            categorical_cols=categorical_cols,
            feateng_cols=feateng_cols,
            units=units,
            feateng_steps=feateng_steps,
            featsel_runs=featsel_runs,
            max_gb=max_gb,
            transformations=transformations,
            apply_pi_theorem=apply_pi_theorem,
            always_return_numpy=always_return_numpy,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    def fit(self, data: np.ndarray | pd.DataFrame, target: np.ndarray | pd.DataFrame):
        set_random_seed(self.random_seed)
        super().fit(data, target)

    def transform(self, data: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
        set_random_seed(self.random_seed)
        data_transformed = super().transform(data)
        return data_transformed

    def fit_transform(
        self, data: np.ndarray | pd.DataFrame, target: np.ndarray | pd.DataFrame
    ) -> np.ndarray | pd.DataFrame:
        set_random_seed(self.random_seed)
        data_transformed = super().fit_transform(data, target)
        return data_transformed


class AutoSpectralIndices(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        problem_type: Literal["regression", "classification"],
        spectral_indices: str | Iterable[str],
        *,
        selector_config: FeatureSelectorConfig = FeatureSelectorConfig(),
        bands_map: dict[str, str] | None = None,
        merge_with_original: bool = True,
    ):
        self.spectral_indices = spectral_indices
        self.selector = feature_selector_factory(
            problem_type=problem_type, config=selector_config
        )
        self.bands_map = bands_map
        self.merge_with_original = merge_with_original

    def fit(self, data: pd.DataFrame, target: pd.Series) -> BaseEstimator:
        df_indices = compute_spectral_indices(
            data=data, spectral_indices=self.spectral_indices, bands_map=self.bands_map
        )
        self.selector.fit(df_indices, target)
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        df_indices = compute_spectral_indices(
            data=data, spectral_indices=self.spectral_indices, bands_map=self.bands_map
        )
        if hasattr(self.selector[1], "k_feature_idx_"):
            columns_select_idx = list(self.selector[1].k_feature_idx_)
            df_indices = df_indices.iloc[:, columns_select_idx]
        else:
            raise AttributeError(
                "The attribute 'k_feature_idx_' does not exist in the selector."
            )
        if self.merge_with_original:
            return pd.concat([data, df_indices], axis=1)
        return df_indices

    def fit_transform(self, data: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        self.fit(data, target)
        return self.transform(data)


class AutoSpectralIndicesClassification(AutoSpectralIndices):
    def __init__(
        self,
        spectral_indices: str | Iterable[str],
        *,
        selector_config: FeatureSelectorConfig = FeatureSelectorConfig(),
        bands_map: dict[str, str] | None = None,
        merge_with_original: bool = True,
    ):
        super().__init__(
            problem_type="classification",
            spectral_indices=spectral_indices,
            selector_config=selector_config,
            bands_map=bands_map,
            merge_with_original=merge_with_original,
        )


class AutoSpectralIndicesRegression(AutoSpectralIndices):
    def __init__(
        self,
        spectral_indices: str | Iterable[str],
        *,
        selector_config: FeatureSelectorConfig = FeatureSelectorConfig(),
        bands_map: dict[str, str] | None = None,
        merge_with_original: bool = True,
    ):
        super().__init__(
            problem_type="regression",
            spectral_indices=spectral_indices,
            selector_config=selector_config,
            bands_map=bands_map,
            merge_with_original=merge_with_original,
        )


# class AutoSpectralIndicesPlusGenerated(BaseEstimator, TransformerMixin):
#     def __init__(
#         self,
#         selector_spectral_indices: Union[
#             "AutoSpectralIndicesPlusGeneratedClassification",
#             "AutoSpectralIndicesPlusGeneratedRegression",
#         ],
#         selector_generated: Union["AutoFeatClassification", "AutoFeatRegression"],
#     ):
#         self.selector_spectral_indices = selector_spectral_indices
#         self.selector_generated = selector_generated

#         self.columns_to_drop: list[str] = []

#     def __repr__(self) -> str:
#         return (
#             f"{self.__class__.__name__}(\n"
#             f"    selector_spectral_indices={self.selector_spectral_indices},\n"
#             f"    selector_generated={self.selector_generated}\n"
#             f")"
#         )

#     def _correlation_analysis(self, data: pd.DataFrame):
#         # Remove highly correlated features
#         # Transform data
#         df_generated = self.selector_generated.transform(data)
#         df_spectral_indices = self.selector_spectral_indices.transform(data)
#         df_merged = pd.concat([df_generated, df_spectral_indices], axis=1)
#         # Calculate correlation between features
#         correlation_matrix = df_merged.corr().abs()
#         upper_triangle = correlation_matrix.where(
#             np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool_)
#         )
#         # Select columns with correlation greater than 0.99
#         self.columns_to_drop = [
#             column
#             for column in upper_triangle.columns
#             if any(upper_triangle[column] > 0.99)
#         ]

#     def fit(self, data: pd.DataFrame, target: pd.Series) -> BaseEstimator:
#         self.selector_generated.fit(data, target)
#         self.selector_spectral_indices.fit(data, target)
#         self._correlation_analysis(data)
#         return self

#     def transform(self, data: pd.DataFrame) -> pd.DataFrame:
#         df_generated = self.selector_generated.transform(data)
#         df_spectral_indices = self.selector_spectral_indices.transform(data)
#         df_merged = pd.concat([df_generated, df_spectral_indices], axis=1)
#         df_merged = df_merged.drop(columns=self.columns_to_drop)
#         return df_merged

#     def fit_transform(self, data: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
#         self.fit(data, target)
#         return self.transform(data)


# class AutoSpectralIndicesPlusGeneratedClassification(AutoSpectralIndicesPlusGenerated):
#     def __init__(self, verbose: int = 0, n_jobs=1, feateng_steps: int = 1, **kwargs):
#         self.verbose = verbose
#         self.n_jobs = n_jobs
#         self.feateng_steps = feateng_steps

#         selector_spectral_indices = AutoSpectralIndicesClassification(
#             verbose=verbose, n_jobs=n_jobs, merge_with_original=False
#         )
#         selector_generated = AutoFeatClassification(
#             verbose=verbose, feateng_steps=feateng_steps
#         )
#         super().__init__(
#             selector_spectral_indices=selector_spectral_indices,
#             selector_generated=selector_generated,
#         )

#     def __repr__(self) -> str:
#         return (
#             f"{self.__class__.__name__}(verbose={self.verbose}, "
#             f"n_jobs={self.n_jobs}, feateng_steps={self.feateng_steps})"
#         )


# class AutoSpectralIndicesPlusGeneratedRegression(AutoSpectralIndicesPlusGenerated):
#     def __init__(self, verbose: int = 0, n_jobs=1, feateng_steps: int = 1, **kwargs):
#         self.verbose = verbose
#         self.n_jobs = n_jobs
#         self.feateng_steps = feateng_steps

#         selector_spectral_indices = AutoSpectralIndicesRegression(
#             verbose=verbose, n_jobs=n_jobs, merge_with_original=False
#         )
#         selector_generated = AutoFeatRegression(
#             verbose=verbose, feateng_steps=feateng_steps
#         )
#         super().__init__(
#             selector_spectral_indices=selector_spectral_indices,
#             selector_generated=selector_generated,
#         )

#     def __repr__(self) -> str:
#         return (
#             f"{self.__class__.__name__}(verbose={self.verbose}, "
#             f"n_jobs={self.n_jobs}, feateng_steps={self.feateng_steps})"
#         )
