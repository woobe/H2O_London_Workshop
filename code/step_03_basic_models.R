# ------------------------------------------------------------------------------
# Step 3: Train Simple Models
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Loading data (same as previous steps)
# ------------------------------------------------------------------------------

# Start and connect to a local H2O cluster
library(h2o)
h2o.init(nthreads = -1)

# Import data from a local CSV file
# Source: https://archive.ics.uci.edu/ml/machine-learning-databases/secom/
secom <- h2o.importFile(path = "./data/secom.csv", destination_frame = "secom")

# Convert Classification to factor
secom$Classification <- as.factor(secom$Classification)


# ------------------------------------------------------------------------------
# Define Targets and Features
# ------------------------------------------------------------------------------

target <- "Classification"
features <- setdiff(colnames(secom), c("ID", "Classification"))

print(target)
print(features)


# ------------------------------------------------------------------------------
# Train H2O models with default value
# ------------------------------------------------------------------------------

# Turn off progress bar (if you want to ...)
# h2o.no_progress()

# GBM
model_gbm <- h2o.gbm(x = features, y = target,
                     training_frame = secom)

# Random Forest
model_drf <- h2o.randomForest(x = features, y = target,
                              training_frame = secom)

# Deep Neural Network
model_dnn <- h2o.deeplearning(x = features, y = target,
                              training_frame = secom)

# Use R / Flow to look at models
print(summary(model_gbm))
print(summary(model_drf))
print(summary(model_dnn))

# Look at variable importance
print(h2o.varimp(model_gbm))
h2o.varimp_plot(model_gbm)


