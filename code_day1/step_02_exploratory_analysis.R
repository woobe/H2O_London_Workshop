# ------------------------------------------------------------------------------
# Step 2: Data Exploration
# ------------------------------------------------------------------------------

# Start and connect to a local H2O cluster
library(h2o)
h2o.init(nthreads = -1)

# Import data from a local CSV file
# Source: https://archive.ics.uci.edu/ml/machine-learning-databases/secom/
secom <- h2o.importFile(path = "./data/secom.csv", destination_frame = "secom")

# Import data from an URL (Optional - just for demo)
# secom <- h2o.importFile(path = "https://github.com/woobe/H2O_London_Workshop/raw/master/data/secom.csv", destination_frame = "secom")

# Basic exploratory analysis
print(dim(secom)) # 1567 x 599
print(summary(secom))
# alternatively, use H2O flow to look at data (localhost:54321)

# Convert Classification to factor
secom$Classification <- as.factor(secom$Classification)

https://github.com/woobe/H2O_London_Workshop/raw/master/data/secom.csv
