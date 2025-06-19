from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

from siapy.optimizers.configs import OptimizeStudyConfig, TabularOptimizerConfig
from siapy.optimizers.optimizers import TabularOptimizer
from siapy.optimizers.parameters import TrialParameters
from siapy.optimizers.scorers import Scorer

# Create a simple model
model = RandomForestRegressor(random_state=42)

# Choose a scoring strategy for model evaluation
# Option 1: Use cross-validation (recommended for small datasets)
scorer = Scorer.init_cross_validator_scorer(
    scoring="neg_mean_squared_error",
    cv=3,  # 3-fold cross-validation
)

# Option 2: Use a hold-out validation set (recommended for larger datasets)
# scorer = Scorer.init_hold_out_scorer(
#     scoring="neg_mean_squared_error",
#     test_size=0.2,  # 20% of data used for validation
# )

# Define trial parameters for hyperparameter optimization
trial_parameters = TrialParameters.from_dict(
    {
        "int_parameters": [
            {"name": "n_estimators", "low": 10, "high": 100},
            {"name": "max_depth", "low": 3, "high": 10},
        ]
    }
)

# Create configuration for optimization
config = TabularOptimizerConfig(
    scorer=scorer,
    trial_parameters=trial_parameters,
    optimize_study=OptimizeStudyConfig(
        n_trials=10,  # Number of trials to run
        timeout=300.0,  # 5 minutes
        n_jobs=1,  # Use a single processor for simplicity
    ),
)

# Mock dataset
X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)

# Create optimizer from dataset
optimizer = TabularOptimizer(model=model, configs=config, X=X, y=y)

# Run optimization
study = optimizer.run()
best_model = optimizer.get_best_model()
