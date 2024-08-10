from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from siapy.utils.general import dict_zip


def assert_estimators_parameters_equal(
    estimator1: BaseEstimator, estimator2: BaseEstimator
) -> bool:
    if len(estimator1.get_params()) != len(estimator2.get_params()):
        return False

    for params_key, params1_val, params2_val in dict_zip(
        estimator1.get_params(), estimator2.get_params()
    ):
        if isinstance(params1_val, BaseEstimator) and isinstance(
            params2_val, BaseEstimator
        ):
            if not assert_estimators_parameters_equal(params1_val, params2_val):
                return False
        elif params1_val != params2_val:
            print(
                f"Values not equal for key: {params_key}",
                f"Value1: {params1_val}, Value2: {params2_val}",
            )
            return False

    return True


def assert_pipelines_parameters_equal(pipeline1: Pipeline, pipeline2: Pipeline) -> bool:
    if len(pipeline1.steps) != len(pipeline2.steps):
        return False

    for step1, step2 in zip(pipeline1.steps, pipeline2.steps):
        if step1[0] != step2[0] or step1[1].__class__ != step2[1].__class__:
            return False

    # Check if the parameters of each step are the same
    for step_name, step1 in pipeline1.named_steps.items():
        step2 = pipeline2.named_steps[step_name]
        if not assert_estimators_parameters_equal(step1, step2):
            return False

    return True
