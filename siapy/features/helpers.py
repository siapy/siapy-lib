from typing import Annotated, Literal

from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import RobustScaler


def feature_selector_factory(
    problem_type: Literal["regression", "classification"],
    *,
    k_features: Annotated[
        int | tuple | str,
        "can be: 'best' - most extensive, (1, n) - check range of features, n - exact number of features",
    ] = (1, 20),
    cv: int = 3,
    forward: Annotated[bool, "selection in forward direction"] = True,
    floating: Annotated[
        bool, "floating algorithm - can go back and remove features once added"
    ] = True,
    verbose: int = 2,
    n_jobs: int = 1,
    pre_dispatch: int | str = "2*n_jobs",
) -> Pipeline:
    """Specific to this particular project and dataset."""
    if problem_type == "regression":
        algo = Ridge()
        scoring = "neg_mean_squared_error"
    elif problem_type == "classification":
        algo = RidgeClassifier()
        scoring = "f1_weighted"
    else:
        raise ValueError(
            f"Invalid problem type: '{problem_type}', possible values are: 'regression' or 'classification'"
        )
    sfs = SequentialFeatureSelector(
        estimator=algo,
        k_features=k_features,  # type: ignore # noqa
        forward=forward,
        floating=floating,
        verbose=verbose,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        pre_dispatch=pre_dispatch,  # type: ignore
    )
    return make_pipeline(RobustScaler(), sfs, memory="cache")
