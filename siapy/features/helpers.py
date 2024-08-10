from typing import Annotated, Literal

from mlxtend.feature_selection import SequentialFeatureSelector  # type: ignore
from pydantic import BaseModel, ConfigDict
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import RobustScaler


class FeatureSelectorConfig(BaseModel):
    k_features: Annotated[
        int | tuple | str,
        "can be: 'best' - most extensive, (1, n) - check range of features, n - exact number of features",
    ] = (1, 20)
    cv: int = 3
    forward: Annotated[bool, "selection in forward direction"] = True
    floating: Annotated[
        bool, "floating algorithm - can go back and remove features once added"
    ] = True
    verbose: int = 2
    n_jobs: int = 1
    pre_dispatch: int | str = "2*n_jobs"

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )


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
    config: Annotated[
        FeatureSelectorConfig | None,
        "If provided, other arguments are overwritten by config values",
    ] = None,
) -> Pipeline:
    if config:
        k_features = config.k_features
        cv = config.cv
        forward = config.forward
        floating = config.floating
        verbose = config.verbose
        n_jobs = config.n_jobs
        pre_dispatch = config.pre_dispatch

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
    return make_pipeline(RobustScaler(), sfs, memory=None)


"""
-- Check plot: performance vs number of features --
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
plot_sfs(self.selector.get_metric_dict(), kind='std_err', figsize=(30, 20))
plt.savefig('selection.png')
plt.close()
"""
