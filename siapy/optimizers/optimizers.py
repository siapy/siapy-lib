from typing import Any

import optuna
from sklearn.base import BaseEstimator, clone

from siapy.core import logger
from siapy.core.exceptions import InvalidInputError
from siapy.core.types import ArrayLike1dType, ArrayLike2dType
from siapy.datasets.schemas import TabularDatasetData
from siapy.optimizers.configs import TabularOptimizerConfig
from siapy.optimizers.parameters import (
    CategoricalParameter,
    FloatParameter,
    IntParameter,
)

__all__ = [
    "TabularOptimizer",
]


class TabularOptimizer:
    def __init__(
        self,
        model: BaseEstimator,
        configs: TabularOptimizerConfig,
        X: ArrayLike2dType,
        y: ArrayLike1dType,
        X_val: ArrayLike2dType | None = None,
        y_val: ArrayLike1dType | None = None,
    ):
        self.model = model
        self.configs = configs
        self.X = X
        self.y = y
        self.X_val = X_val
        self.y_val = y_val

        self._study: optuna.study.Study | None = None

    @classmethod
    def from_tabular_dataset_data(
        cls,
        model: BaseEstimator,
        configs: TabularOptimizerConfig,
        data: TabularDatasetData,
        data_val: TabularDatasetData | None = None,
    ) -> "TabularOptimizer":
        signals = data.signatures.signals.df
        target = data.target
        signals_val = data_val.signatures.signals.df if data_val else None
        target_val = data_val.target if data_val else None

        if target is None:
            raise InvalidInputError(
                input_value=target,
                message="Target data is required for optimization.",
            )
        if signals_val is not None and target_val is None:
            raise InvalidInputError(
                input_value={
                    "signals_val": signals_val,
                    "target_val": target_val,
                },
                message=(
                    "If validation data (data_val) is provided, "
                    "validation targets (data_val.target) must also be provided."
                ),
            )
        return cls(
            model=model,
            configs=configs,
            X=signals,
            y=target.value,
            X_val=signals_val,
            y_val=target_val.value if target_val else None,
        )

    @property
    def study(self) -> optuna.study.Study | None:
        return self._study

    @property
    def best_trial(self) -> optuna.trial.FrozenTrial | None:
        return self.study.best_trial if self.study else None

    def run(self) -> optuna.study.Study:
        study = optuna.create_study(**self.configs.create_study.model_dump())
        study.optimize(
            self.objective,
            **self.configs.optimize_study.model_dump(),
        )
        if self.best_trial:
            logger.info("Best scoring metric: %s", self.best_trial.value)
            logger.info("Best hyperparameters found were: %s", self.best_trial.params)
        self._study = study
        return study

    def get_best_model(self) -> BaseEstimator:
        if self.best_trial is None:
            raise InvalidInputError(
                input_value="None",
                message="Study is not available for model refitting.",
            )

        best_model = clone(self.model)
        best_model.set_params(**self.best_trial.params)
        best_model.fit(self.X, self.y)
        return best_model

    def objective(self, trial: optuna.trial.Trial) -> float:
        params = self._trial_params(trial)
        self.model = clone(self.model)
        self.model.set_params(**params)
        score = self.scorer()
        return score

    def _trial_params(self, trial: optuna.trial.Trial) -> dict[str, Any]:
        if self.configs.trial_parameters is None:
            raise InvalidInputError(
                input_value="None",
                message="Trial parameters are not defined. "
                "Add trial_parameters to configs or implement your custom objective function.",
            )

        params: dict[str, Any] = {}
        p: IntParameter | FloatParameter | CategoricalParameter
        for p in self.configs.trial_parameters.int_parameters:
            params[p.name] = trial.suggest_int(
                name=p.name,
                low=p.low,
                high=p.high,
                step=p.step,
                log=p.log,
            )
        for p in self.configs.trial_parameters.float_parameters:
            params[p.name] = trial.suggest_float(
                name=p.name,
                low=p.low,
                high=p.high,
                step=p.step,
                log=p.log,
            )
        for p in self.configs.trial_parameters.categorical_parameters:
            params[p.name] = trial.suggest_categorical(
                name=p.name,
                choices=p.choices,
            )
        return params

    def scorer(self) -> float:
        if self.configs.scorer is None:
            raise InvalidInputError(
                input_value=self.configs.scorer,
                message="Scorer is not defined. Add scorer to configs or implement your custom scorer.",
            )
        return self.configs.scorer(
            model=self.model,
            X=self.X,
            y=self.y,
            X_val=self.X_val,
            y_val=self.y_val,
        )
