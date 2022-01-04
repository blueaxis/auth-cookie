# TFDF Documentation: https://www.tensorflow.org/decision_forests/

# Note regarding tfdf module
# It cannot be installed using Windows pip
# Either use Linux or WSL
import tensorflow_decision_forests as tfdf
import pandas as pd

# TODO: Implement all helper functions
# Helper functions for feature identification
# *******************************************
COOKIE_ALPHABET = "abdefghijklmnqrstuvxyzABDEFGHIJKLMNQRSTUVXYZ0123456789!#$%&'()*+-./:<>?@[]^_`{|}~"
N = len(COOKIE_ALPHABET)


def index_of_coincidence(c_value):
    for c in COOKIE_ALPHABET:
        pass
    return 0


# *******************************************


# Read dataset from CSV file
# Remove unused columns
col_names = ["Website", "ID", "Name", "Value", "Domain", "Path", "Secure", "Expiry", "HTTP-Only", "JavaScript", "Class"]
col_used = ["Website", "Name", "Value", "Secure", "Expiry", "HTTP-Only", "JavaScript", "Class"]
dataset = pd.read_csv("data/cookies.csv", names=col_names, usecols=col_used)

# Split dataset into training and test set (8:2) ratio
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# No idea if this is needed
tfdf_train = tfdf.keras.pd_dataframe_to_tf_dataset(train_dataset, label="Class")
tfdf_test = tfdf.keras.pd_dataframe_to_tf_dataset(test_dataset, label="Class")

# Initialize then train model
RF_model = tfdf.keras.RandomForestModel()
RF_model.fit(tfdf_train)

# Evaluate model
# Metrics are available here: https://www.tensorflow.org/api_docs/python/tf/keras/metrics
# Accuracy is only used for testing purposes
RF_model.compile(metrics=["accuracy"])
evaluation = RF_model.evaluate(tfdf_test, return_dict=True)

for name, value in evaluation.items():
  print(f"{name}: {value:.4f}")

# TODO: Understand what the summary prints
# RF_model.summary()
