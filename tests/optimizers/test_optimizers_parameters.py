import pytest

from siapy.optimizers.parameters import (
    CategoricalParameter,
    FloatParameter,
    IntParameter,
    TrialParameters,
)


def test_init():
    float_params = [FloatParameter(name="fp1", low=0.0, high=1.0)]
    int_params = [IntParameter(name="ip1", low=0, high=10)]
    cat_params = [CategoricalParameter(name="cp1", choices=[True, False])]
    trial_params = TrialParameters(
        float_parameters=float_params,
        int_parameters=int_params,
        categorical_parameters=cat_params,
    )
    assert trial_params.float_parameters == float_params
    assert trial_params.int_parameters == int_params
    assert trial_params.categorical_parameters == cat_params


def test_init_defaults():
    trial_params = TrialParameters()
    assert trial_params.float_parameters == []
    assert trial_params.int_parameters == []
    assert trial_params.categorical_parameters == []
    trial_params = TrialParameters.from_dict({})
    assert trial_params.float_parameters == []
    assert trial_params.int_parameters == []
    assert trial_params.categorical_parameters == []


def test_from_dict():
    parameters_dict = {
        "float_parameters": [{"name": "fp1", "low": 0.0, "high": 1.0}],
        "int_parameters": [{"name": "ip1", "low": 0, "high": 10}],
        "categorical_parameters": [{"name": "cp1", "choices": [True, False]}],
    }

    trial_params = TrialParameters.from_dict(parameters_dict)

    assert len(trial_params.float_parameters) == 1
    assert trial_params.float_parameters[0].name == "fp1"
    assert trial_params.float_parameters[0].low == pytest.approx(0.0, rel=1e-2)
    assert trial_params.float_parameters[0].high == pytest.approx(1, rel=1e-2)

    assert len(trial_params.int_parameters) == 1
    assert trial_params.int_parameters[0].name == "ip1"
    assert trial_params.int_parameters[0].low == 0
    assert trial_params.int_parameters[0].high == 10

    assert len(trial_params.categorical_parameters) == 1
    assert trial_params.categorical_parameters[0].name == "cp1"
    assert trial_params.categorical_parameters[0].choices == [True, False]
