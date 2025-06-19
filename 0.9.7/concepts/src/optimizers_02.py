from siapy.optimizers.parameters import (
    CategoricalParameter,
    FloatParameter,
    IntParameter,
    TrialParameters,
)

# Define individual parameters
n_estimators_param = IntParameter(name="n_estimators", low=10, high=200, step=10)
learning_rate_param = FloatParameter(name="learning_rate", low=0.01, high=0.3, log=True)
algorithm_param = CategoricalParameter(name="algorithm", choices=["auto", "ball_tree", "kd_tree", "brute"])

# Create trial parameters
trial_parameters = TrialParameters(
    int_parameters=[n_estimators_param],
    float_parameters=[learning_rate_param],
    categorical_parameters=[algorithm_param],
)

# Alternative: create from dictionary
trial_parameters_from_dict = TrialParameters.from_dict(
    {
        "int_parameters": [{"name": "n_estimators", "low": 10, "high": 200, "step": 10}],
        "float_parameters": [{"name": "learning_rate", "low": 0.01, "high": 0.3, "log": True}],
        "categorical_parameters": [{"name": "algorithm", "choices": ["auto", "ball_tree", "kd_tree", "brute"]}],
    }
)
