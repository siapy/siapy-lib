from siapy.optimizers.scorers import Scorer

# Classification hold-out scorer
holdout_scorer = Scorer.init_hold_out_scorer(scoring="accuracy", test_size=0.2)

# Regression hold-out scorer
regression_scorer = Scorer.init_hold_out_scorer(scoring="neg_mean_absolute_error", test_size=0.25)
