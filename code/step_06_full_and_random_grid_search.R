# ------------------------------------------------------------------------------
# Step 6: Tuning Models via Early Stopping & Full/Random Grid Search
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Loading and preparing data (same as previous steps)
# ------------------------------------------------------------------------------

# Start and connect to a local H2O cluster
library(h2o)
h2o.init(nthreads = -1)

# Import data from a local CSV file
secom <- h2o.importFile(path = "./data/secom.csv", destination_frame = "secom")

# Convert Classification to factor
secom$Classification <- as.factor(secom$Classification)

# Define Targets and Features
target <- "Classification"
features <- setdiff(colnames(secom), c("ID", "Classification"))

# Split
secom_splits <- h2o.splitFrame(data = secom, ratios = 0.6, seed = 1234)
secom_train <- secom_splits[[1]]  # 882 : 62 ... 7% of 1
secom_valid  <- secom_splits[[2]]  # 581 : 42 ... 7% of 1


# ------------------------------------------------------------------------------
# Train H2O models with FULL grid search
# ------------------------------------------------------------------------------

# Define parameters for grid search
param_dnn <- list(
  activation = c("Tanh", "Rectifier"),
  hidden = list(c(50,50), c(50,50,50), c(100,100)),
  balance_classes = c(TRUE, FALSE)
)

# DNN with early stopping and FULL grid search
full_grid_dnn <- h2o.grid(

  # Core parameters for model training
  x = features,
  y = target,
  training_frame = secom_train,
  validation_frame = secom_valid,
  epochs = 100,

  # Parameters for grid search
  grid_id = "full_grid_dnn",
  hyper_params = param_dnn,
  algorithm = "deeplearning",

  # Parameters for early stopping
  stopping_metric = "logloss",
  stopping_rounds = 10

)


# ------------------------------------------------------------------------------
# Extract the model and evaluate model performance on unseen data
# ------------------------------------------------------------------------------

# Sort models by metric "logloss"
full_grid_sort <- h2o.getGrid("full_grid_dnn", sort_by = "logloss", decreasing = FALSE)
print(full_grid_sort)

# Extract the best model
model_ids <- full_grid_sort@model_ids
best_dnn <- h2o.getModel(model_ids[[1]])



# ------------------------------------------------------------------------------
# Train H2O models with early stopping and RANDOM grid search
# ------------------------------------------------------------------------------

# Define parameters for grid search
param_dnn <- list(
  activation = c("Tanh", "Rectifier"),
  hidden = list(c(50,50), c(50,50,50), c(100,100)),
  balance_classes = c(TRUE, FALSE)
)

# Define criteria for random grid search
search_criteria <- list(
  strategy = "RandomDiscrete",   # Ask H2O to run random grid search
  max_models = 5,                # e.g. only want to test 5 random models
  max_runtime_secs = 600,        # e.g. only have 600 sec to run this
  seed = 1234                    # reproducible random parameters combinations
)

# DNN with early stopping and RANDOM grid search
random_grid_dnn <- h2o.grid(
  
  # Core parameters for model training
  x = features,
  y = target,
  training_frame = secom_train,
  validation_frame = secom_valid,
  epochs = 100,
  
  # Parameters for grid search
  grid_id = "random_grid_dnn",
  hyper_params = param_dnn,
  algorithm = "deeplearning",
  search_criteria = search_criteria, # <- added this for Random Grid Search
  
  # Parameters for early stopping
  stopping_metric = "logloss",
  stopping_rounds = 10
  
)

# Sort models by metric "logloss"
random_grid_sort <- h2o.getGrid("random_grid_dnn",
                             sort_by = "logloss", decreasing = FALSE)
print(random_grid_sort)
