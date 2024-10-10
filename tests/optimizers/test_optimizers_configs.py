import optuna
import pytest

from siapy.optimizers.configs import (
    CreateStudyConfig,
    OptimizeStudyConfig,
    TabularOptimizerConfig,
)
from siapy.optimizers.parameters import TrialParameters
from siapy.optimizers.scorers import Scorer


def test_create_study_config_defaults():
    config = CreateStudyConfig()
    assert config.storage is None
    assert config.sampler is None
    assert config.pruner is None
    assert config.study_name is None
    assert config.direction == "minimize"
    assert config.load_if_exists is False


def test_optimize_study_config_defaults():
    config = OptimizeStudyConfig()
    assert config.n_trials is None
    assert config.timeout is None
    assert config.n_jobs == -1
    assert config.catch == ()
    assert config.callbacks is None
    assert config.gc_after_trial is False
    assert config.show_progress_bar is True


def test_tabular_optimizer_config_defaults():
    config = TabularOptimizerConfig()
    assert isinstance(config.create_study, CreateStudyConfig)
    assert isinstance(config.optimize_study, OptimizeStudyConfig)
    assert config.scorer is None
    assert config.trial_parameters is None


def test_create_study_config_custom_values():
    config = CreateStudyConfig(
        storage="sqlite:///example.db",
        sampler=optuna.samplers.RandomSampler(),
        pruner=optuna.pruners.MedianPruner(),
        study_name="test_study",
        direction="maximize",
        load_if_exists=True,
    )
    assert config.storage == "sqlite:///example.db"
    assert isinstance(config.sampler, optuna.samplers.RandomSampler)
    assert isinstance(config.pruner, optuna.pruners.MedianPruner)
    assert config.study_name == "test_study"
    assert config.direction == "maximize"
    assert config.load_if_exists is True


def test_optimize_study_config_custom_values():
    config = OptimizeStudyConfig(
        n_trials=100,
        timeout=3600.0,
        n_jobs=4,
        catch=(ValueError,),
        callbacks=[lambda study, trial: None],
        gc_after_trial=True,
        show_progress_bar=False,
    )
    assert config.n_trials == 100
    assert config.timeout == pytest.approx(3600)
    assert config.n_jobs == 4
    assert config.catch
    assert len(config.callbacks) == 1
    assert config.gc_after_trial is True
    assert config.show_progress_bar is False


def test_tabular_optimizer_config_custom_values():
    scorer = Scorer.init_cross_validator_scorer()
    trial_parameters = TrialParameters()
    config = TabularOptimizerConfig(
        create_study=CreateStudyConfig(study_name="custom_study"),
        optimize_study=OptimizeStudyConfig(n_trials=50),
        scorer=scorer,
        trial_parameters=trial_parameters,
    )
    assert config.create_study.study_name == "custom_study"
    assert config.optimize_study.n_trials == 50
    assert config.scorer == scorer
    assert config.trial_parameters == trial_parameters
