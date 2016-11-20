# ------------------------------------------------------------------------------
# Step 4: Evaluate Models
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
# Method 1: Split data into training / test
# ------------------------------------------------------------------------------

# Split
# i.e. using 60% of data for training and 40% for test
secom_splits <- h2o.splitFrame(data = secom, ratios = 0.6, seed = 1234)
secom_train <- secom_splits[[1]]    # optional step
secom_test  <- secom_splits[[2]]    # optional step

# Check ratio
summary(secom_train$Classification) # 882 : 62 ... % of 1 = 0.07029478
summary(secom_test$Classification) # 581 : 42 ... % of 1 = 0.07228916

# Train a simple Deep Learning model using 60% of data
model_dnn1 <- h2o.deeplearning(x = features, y = target,
                               training_frame = secom_train)

# Evaluate model performance on unseen (40%) data
h2o.performance(model_dnn1, newdata = secom_test)


# ------------------------------------------------------------------------------
# Method 2: K-fold Cross-Validation
# ------------------------------------------------------------------------------

# Train a simple Deep Learning model using 100% of data with n-fold CV
model_dnn2 <- h2o.deeplearning(x = features, y = target,
                               training_frame = secom,
                               nfolds = 5,
                               seed = 1234,
                               fold_assignment = "Stratified")

# Look at the evaluation results on n-fold CV
print(model_dnn2)

