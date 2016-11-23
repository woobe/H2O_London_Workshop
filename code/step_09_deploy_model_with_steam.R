# ------------------------------------------------------------------------------
# Step 9: Deploying a model using H2O Steam
# See http://docs.h2o.ai/steam/latest-stable/index.html
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Build a simple classification model using iris dataset
# ------------------------------------------------------------------------------

# Start and connect to a local H2O cluster
library(h2o)
h2o.init(nthreads = -1)

# Import data from a R data frame
data(iris)
d_iris <- as.h2o(iris)

# Quick look
head(d_iris)
summary(d_iris)


# Define Targets and Features
target <- "Species"
features <- setdiff(colnames(d_iris), c("Species"))


# ------------------------------------------------------------------------------
# Train a H2O Model
# ------------------------------------------------------------------------------

# Train a Deep Learning model
model <- h2o.deeplearning(x = features,
                          y = target,
                          model_id = "iris_deep_learning",
                          training_frame = d_iris)


# ------------------------------------------------------------------------------
# Use Steam to deploy model
# ------------------------------------------------------------------------------

# Live demo time
# See http://docs.h2o.ai/steam/latest-stable/index.html

# Basic Steps
# 1. In terminal, go to steam folder
# 2. [enter] java -jar var/master/assets/jetty-runner.jar var/master/assets/ROOT.war
# 3. In another terminal, go to the same steam folder
# 4. [enter] ./steam serve master --superuser-name=superuser --superuser-password=superuser
# 5. go to steam web interface (localhost:9000)
# 6. Create a project and continue to deploy a model

