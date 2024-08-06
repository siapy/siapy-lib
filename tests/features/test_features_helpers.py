import pytest
from sklearn.pipeline import Pipeline

from siapy.features.helpers import feature_selector_factory


def test_regression_pipeline():
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


def test_classification_pipeline():
    pipeline = feature_selector_factory(problem_type="classification")
    assert isinstance(pipeline, Pipeline)
    assert pipeline.named_steps["sequentialfeatureselector"].scoring == "f1_weighted"
    assert (
        pipeline.named_steps["sequentialfeatureselector"].estimator.__class__.__name__
        == "RidgeClassifier"
    )


def test_invalid_problem_type():
    with pytest.raises(ValueError):
        feature_selector_factory(problem_type="invalid")


def test_custom_args():
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
