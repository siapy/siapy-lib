from sklearn.model_selection import RepeatedKFold

from siapy.optimizers.scorers import Scorer

# Basic cross-validation scorer
cv_scorer = Scorer.init_cross_validator_scorer(scoring="accuracy", cv=5)

# Advanced cross-validation with custom CV strategy
repeated_cv = RepeatedKFold(n_splits=3, n_repeats=5, random_state=42)
advanced_scorer = Scorer.init_cross_validator_scorer(
    scoring="f1_weighted",
    cv=repeated_cv,
    n_jobs=-1,  # Use all processors
)

# Cross-validation with repeated splits (string shortcut)
repeated_scorer = Scorer.init_cross_validator_scorer(scoring="neg_mean_squared_error", cv="RepeatedKFold")
