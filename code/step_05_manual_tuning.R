# ------------------------------------------------------------------------------
# Step 5: Tuning Models with Manual Tweaks
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


# ------------------------------------------------------------------------------
# Train H2O models with default / manual settings
# ------------------------------------------------------------------------------

# Check out all parameters
# ?h2o.gbm
# ?h2o.deeplearning
# ?h2o.randomForest

# Deep Learning model with CV and default value
model_dnn1 <- h2o.deeplearning(x = features, 
                               y = target,
                               training_frame = secom,
                               nfolds = 5,
                               seed = 1234,
                               fold_assignment = "Stratified")
print(model_dnn1)

# Deep Learning model with manual settings
# ?h2o.deeplearning
model_dnn2 <- h2o.deeplearning(x = features, 
                               y = target,
                               training_frame = secom,
                               nfolds = 5,
                               seed = 1234,
                               fold_assignment = "Stratified",
                               
                               # Manual tweaks
                               activation = "RectifierWithDropout",
                               balance_classes = TRUE,
                               hidden = c(50, 50, 50),
                               epochs = 100)
print(model_dnn2)

# Use R / Flow to look at models
print(model_dnn1)
print(model_dnn2)

# ------------------------------------------------------------------------------
# Making predictions
# ------------------------------------------------------------------------------

yhat <- h2o.predict(model_dnn2, secom)
print(head(yhat, 40))
print(summary(yhat))

