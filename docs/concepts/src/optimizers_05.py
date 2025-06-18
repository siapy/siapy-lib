import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from siapy.datasets.schemas import RegressionTarget
from siapy.datasets.tabular import TabularDataset
from siapy.entities import Shape, SpectralImage, SpectralImageSet
from siapy.optimizers.configs import CreateStudyConfig, OptimizeStudyConfig, TabularOptimizerConfig
from siapy.optimizers.optimizers import TabularOptimizer
from siapy.optimizers.parameters import TrialParameters
from siapy.optimizers.scorers import Scorer

# ========================================================================
# STEP 1: Create Mock Spectral Data for Demonstration
# ========================================================================

# Initialize random number generator for reproducible results
rng = np.random.default_rng(seed=42)

# Define a rectangular region of interest (ROI) within our images
# This simulates selecting a specific area for analysis (e.g., crop field, water body)
rectangle = Shape.from_rectangle(x_min=10, y_min=15, x_max=20, y_max=30)

# Generate two mock spectral images with 4 spectral bands each
# In real applications, these would be actual hyperspectral/multispectral images
# Shape: (height=50, width=50, bands=4)
image1 = SpectralImage.from_numpy(rng.random((50, 50, 4)))
image2 = SpectralImage.from_numpy(rng.random((50, 50, 4)))

# Attach the same region of interest to both images
# This ensures we analyze the same spatial area across all images
image1.geometric_shapes.append(rectangle)
image2.geometric_shapes.append(rectangle)

# Combine individual images into a spectral image set
# This allows us to process multiple images together
image_set = SpectralImageSet([image1, image2])

# ========================================================================
# STEP 2: Convert Spectral Images to Tabular Dataset
# ========================================================================

# Create a tabular dataset from our spectral image set
# This extracts pixel values from the ROI and organizes them in a table format
dataset = TabularDataset(image_set)

# Process the image data to extract spectral signatures from the defined ROI
# This step converts spatial pixel data into a structured format for analysis
dataset.process_image_data()

# Generate the actual dataset with individual pixel signatures (not averaged)
# Setting mean_signatures=False keeps each pixel as a separate data point
dataset_data = dataset.generate_dataset_data(mean_signatures=False)

# Display the structure of our extracted data
# This shows how spectral values are organized in tabular format
print(f"Dataframe: {dataset_data.to_dataframe().head()}")

# ========================================================================
# STEP 3: Create Target Variables for Regression
# ========================================================================

# Create synthetic regression target values for demonstration
# In real applications, these would be actual measurements (e.g., soil moisture, vegetation health)
values = pd.Series(rng.random(len(dataset_data.signatures.signals)))
target = RegressionTarget(value=values)

# Attach the target values to our dataset
# This creates a complete supervised learning dataset
dataset_data = dataset_data.set_attributes(target=target)

# ========================================================================
# STEP 4: Setup Machine Learning Model, Parameters, Scorer and Configs
# ========================================================================

# Choose a machine learning model for regression
model = RandomForestRegressor(random_state=42)

# Define the hyperparameter search space for optimization
# We'll optimize the number of estimators (trees) in the forest
# The optimizer will try values from 10 to 100 in steps of 10
trial_parameters = TrialParameters.from_dict(
    {
        "int_parameters": [
            {"name": "n_estimators", "low": 10, "high": 100, "step": 10},
        ],
    }
)

# Configure the scoring method for model evaluation
# We use cross-validation with negative mean squared error (higher is better)
# CV=3 means 3-fold cross-validation for more robust performance estimates
scorer = Scorer.init_cross_validator_scorer(scoring="neg_mean_squared_error", cv=3)

# Create comprehensive optimizer configuration
config = TabularOptimizerConfig(
    scorer=scorer,  # How to evaluate model performance
    trial_parameters=trial_parameters,  # What hyperparameters to optimize
    optimize_study=OptimizeStudyConfig(
        n_trials=50,  # Try 50 different hyperparameter combinations
        n_jobs=-1,  # Use all available CPU cores for parallel processing
    ),
    create_study=CreateStudyConfig(
        direction="maximize",  # We want to maximize the score (minimize error)
        study_name="RandomForestOptimizationStudy",  # Name for tracking this optimization
    ),
)

# ========================================================================
# STEP 5: Execute Optimization and Retrieve Results
# ========================================================================

# Create the optimizer instance with our model, configuration, and data
# This sets up everything needed for hyperparameter optimization
optimizer = TabularOptimizer.from_tabular_dataset_data(model=model, configs=config, data=dataset_data)

# Run the optimization process
# This will test 50 different hyperparameter combinations and find the best one
study = optimizer.run()

# Get the best performing model with optimized hyperparameters
best_model = optimizer.get_best_model()

# Display the optimization results
print(f"Best trial: {optimizer.best_trial}")
if optimizer.best_trial is not None:
    print(f"Best parameters: {optimizer.best_trial.params}")
    print(f"Best score: {optimizer.best_trial.value}")
else:
    print("No best trial found - optimization may have failed")
