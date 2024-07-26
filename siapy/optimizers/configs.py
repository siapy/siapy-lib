from typing import Callable, Iterable, Literal

import optuna
from pydantic import BaseModel, ConfigDict

from siapy.optimizers.parameters import TrialParameters
from siapy.optimizers.scorers import Scorer


class CreateStudyConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    storage: str | optuna.storages.BaseStorage | None = None
    sampler: optuna.samplers.BaseSampler | None = None
    pruner: optuna.pruners.BasePruner | None = None
    study_name: str | None = None
    direction: Literal["maximize", "minimize"] | optuna.study.StudyDirection | None = (
        "minimize"
    )
    load_if_exists: bool = False


class OptimizeStudyConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    n_trials: int | None = None
    timeout: float | None = None
    n_jobs: int = 1
    catch: Iterable[type[Exception]] | type[Exception] = ()
    callbacks: (
        list[Callable[[optuna.study.Study, optuna.trial.FrozenTrial], None]] | None
    ) = None
    gc_after_trial: bool = False
    show_progress_bar: bool = True


class TabularOptimizerConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    create_study: CreateStudyConfig = CreateStudyConfig()
    optimize_study: OptimizeStudyConfig = OptimizeStudyConfig()
    scorer: Scorer | None = None
    trial_parameters: TrialParameters | None = None
