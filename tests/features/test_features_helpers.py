import pytest
from sklearn.pipeline import Pipeline

from siapy.features.helpers import FeatureSelectorConfig, feature_selector_factory
from tests.utils import assert_pipelines_parameters_equal


def test_feature_selector_factory_regression_pipeline():
    pipeline = feature_selector_factory(problem_type="regression")
    assert isinstance(pipeline, Pipeline)
    assert (
        pipeline.named_steps["sequentialfeatureselector"].scoring
        == "neg_mean_squared_error"
    )
    assert (
        pipeline.named_steps["sequentialfeatureselector"].estimator.__class__.__name__
        == "Ridge"
    )


def test_feature_selector_factory_classification_pipeline():
    pipeline = feature_selector_factory(problem_type="classification")
    assert isinstance(pipeline, Pipeline)
    assert pipeline.named_steps["sequentialfeatureselector"].scoring == "f1_weighted"
    assert (
        pipeline.named_steps["sequentialfeatureselector"].estimator.__class__.__name__
        == "RidgeClassifier"
    )


def test_feature_selector_factory_invalid_problem_type():
    with pytest.raises(ValueError):
        feature_selector_factory(problem_type="invalid")


def test_feature_selector_factory_custom_args():
    pipeline = feature_selector_factory(problem_type="regression", k_features=5)
    assert pipeline.named_steps["sequentialfeatureselector"].k_features == 5

    pipeline = feature_selector_factory(problem_type="regression", cv=5)
    assert pipeline.named_steps["sequentialfeatureselector"].cv == 5

    pipeline = feature_selector_factory(problem_type="regression", forward=False)
    assert not pipeline.named_steps["sequentialfeatureselector"].forward

    pipeline = feature_selector_factory(problem_type="regression", floating=False)
    assert not pipeline.named_steps["sequentialfeatureselector"].floating

    pipeline = feature_selector_factory(problem_type="regression", verbose=0)
    assert pipeline.named_steps["sequentialfeatureselector"].verbose == 0

    pipeline = feature_selector_factory(problem_type="regression", n_jobs=2)
    assert pipeline.named_steps["sequentialfeatureselector"].n_jobs == 2


def test_feature_selector_factory_config_vs_args():
    config = FeatureSelectorConfig()
    pipeline_reg_with_args = feature_selector_factory(problem_type="regression")
    pipeline_reg_with_config = feature_selector_factory(
        problem_type="regression", config=config
    )
    pipeline_clf_with_args = feature_selector_factory(problem_type="classification")
    pipeline_clf_with_config = feature_selector_factory(
        problem_type="classification", config=config
    )

    assert assert_pipelines_parameters_equal(
        pipeline_reg_with_args, pipeline_reg_with_config
    )
    assert assert_pipelines_parameters_equal(
        pipeline_clf_with_args, pipeline_clf_with_config
    )
    assert not assert_pipelines_parameters_equal(
        pipeline_reg_with_args, pipeline_clf_with_args
    )
    assert not assert_pipelines_parameters_equal(
        pipeline_reg_with_config, pipeline_clf_with_config
    )

    config2 = FeatureSelectorConfig(cv=2)
    pipeline_reg_with_config = feature_selector_factory(
        problem_type="regression", config=config2
    )
    pipeline_clf_with_config = feature_selector_factory(
        problem_type="classification", config=config2
    )

    assert not assert_pipelines_parameters_equal(
        pipeline_reg_with_args, pipeline_reg_with_config
    )
    assert not assert_pipelines_parameters_equal(
        pipeline_clf_with_args, pipeline_clf_with_config
    )
