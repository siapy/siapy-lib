import optuna
import pytest
from sklearn.base import BaseEstimator
from sklearn.svm import SVR

from siapy.datasets.schemas import RegressionTarget
from siapy.optimizers.configs import OptimizeStudyConfig, TabularOptimizerConfig
from siapy.optimizers.optimizers import TabularOptimizer
from siapy.optimizers.parameters import TrialParameters
from siapy.optimizers.scorers import Scorer


def test_tabular_optimizer_from_tabular_dataset_data_valid(spectral_tabular_dataset):
    data = spectral_tabular_dataset.dataset_data.model_copy()
    data.target = RegressionTarget(name="test", value=data.pixels["u"])
    model = SVR()
    configs = TabularOptimizerConfig()
    optimizer = TabularOptimizer.from_tabular_dataset_data(
        model=model, configs=configs, data=data
    )
    assert isinstance(optimizer, TabularOptimizer)
    assert optimizer.model == model
    assert optimizer.configs == configs
    assert optimizer.X.equals(data.signals)
    assert list(optimizer.y) == list(data.target.value)


def test_tabular_optimizer_from_tabular_dataset_data_invalid(spectral_tabular_dataset):
    data = spectral_tabular_dataset.dataset_data
    model = SVR()
    configs = TabularOptimizerConfig()
    with pytest.raises(ValueError, match="Target data is required for optimization"):
        TabularOptimizer.from_tabular_dataset_data(
            model=model, configs=configs, data=data
        )


def test_tabular_optimizer_with_validation_data_without_targets(
    spectral_tabular_dataset,
):
    data = spectral_tabular_dataset.dataset_data.model_copy()
    data_val = spectral_tabular_dataset.dataset_data.model_copy()
    target = RegressionTarget(name="test", value=data.pixels["u"])
    data.target = target
    model = SVR()
    configs = TabularOptimizerConfig()
    with pytest.raises(ValueError, match="validation targets"):
        TabularOptimizer.from_tabular_dataset_data(
            model=model, configs=configs, data=data, data_val=data_val
        )


def test_tabular_optimizer_with_validation_data(spectral_tabular_dataset):
    data = spectral_tabular_dataset.dataset_data.model_copy()
    data_val = spectral_tabular_dataset.dataset_data.model_copy()
    data.target = RegressionTarget(name="test", value=data.pixels["u"])
    data_val.target = RegressionTarget(name="test_val", value=data_val.pixels["v"])
    model = SVR()
    configs = TabularOptimizerConfig()
    optimizer = TabularOptimizer.from_tabular_dataset_data(
        model=model, configs=configs, data=data, data_val=data_val
    )
    assert isinstance(optimizer, TabularOptimizer)
    assert optimizer.model == model
    assert optimizer.configs == configs
    assert optimizer.X.equals(data.signals)
    assert list(optimizer.y) == list(data.target.value)
    assert optimizer.X_val.equals(data_val.signals)
    assert list(optimizer.y_val) == list(data_val.target.value)


def test_tabular_optimizer_run(spectral_tabular_dataset):
    data = spectral_tabular_dataset.dataset_data.model_copy()
    data.signals = data.signals.iloc[:, :10]
    data.target = RegressionTarget(name="test", value=data.pixels["u"])
    model = SVR()
    study_config = OptimizeStudyConfig(n_trials=3, timeout=3000)
    trial_parameters = TrialParameters.from_dict(
        {"int_parameters": [{"name": "C", "low": 1, "high": 100}]}
    )
    scorer = Scorer.init_hold_out_scorer(test_size=0.34, scoring="max_error")
    configs = TabularOptimizerConfig(
        trial_parameters=trial_parameters, scorer=scorer, optimize_study=study_config
    )
    optimizer = TabularOptimizer.from_tabular_dataset_data(
        model=model, configs=configs, data=data
    )
    assert optimizer.study is None
    assert optimizer.best_trial is None
    study = optimizer.run()
    assert isinstance(study, optuna.study.Study)
    best_model = optimizer.get_best_model()
    assert isinstance(best_model, BaseEstimator)
    all_params = best_model.get_params()
    assert {"C": all_params["C"]} == optimizer.best_trial.params
    assert optimizer.best_trial.value == 0


def test_tabular_optimizer_get_best_model_no_best_trial(spectral_tabular_dataset):
    data = spectral_tabular_dataset.dataset_data.model_copy()
    data.target = RegressionTarget(name="test", value=data.pixels["u"])
    model = SVR()
    configs = TabularOptimizerConfig()
    optimizer = TabularOptimizer.from_tabular_dataset_data(
        model=model, configs=configs, data=data
    )
    with pytest.raises(ValueError, match="Study is not available for model refitting."):
        optimizer.get_best_model()


def test_tabular_optimizer_trial_params_no_trial_parameters(spectral_tabular_dataset):
    data = spectral_tabular_dataset.dataset_data.model_copy()
    data.target = RegressionTarget(name="test", value=data.pixels["u"])
    model = SVR()
    configs = TabularOptimizerConfig()
    optimizer = TabularOptimizer.from_tabular_dataset_data(
        model=model, configs=configs, data=data
    )
    trial = optuna.trial.FixedTrial({})
    with pytest.raises(ValueError, match="Trial parameters are not defined."):
        optimizer._trial_params(trial)


def test_tabular_optimizer_scorer_no_scorer_defined(spectral_tabular_dataset):
    data = spectral_tabular_dataset.dataset_data.model_copy()
    data.target = RegressionTarget(name="test", value=data.pixels["u"])
    model = SVR()
    configs = TabularOptimizerConfig()
    optimizer = TabularOptimizer.from_tabular_dataset_data(
        model=model, configs=configs, data=data
    )
    with pytest.raises(ValueError, match="Scorer is not defined."):
        optimizer.scorer()
