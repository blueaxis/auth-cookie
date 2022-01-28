# TFDF Documentation: https://www.tensorflow.org/decision_forests/

# Note regarding tfdf module
# It cannot be installed using Windows pip
# Either use Linux or WSL
import tensorflow_decision_forests as tfdf
import tensorflow as tf
import pandas as pd

# Read dataset from CSV file
dataset = pd.read_csv("data/features.csv")

# Split dataset into training and test set (8:2) ratio
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# No idea if this is needed
tfdf_train = tfdf.keras.pd_dataframe_to_tf_dataset(train_dataset, label="Class")
tfdf_test = tfdf.keras.pd_dataframe_to_tf_dataset(test_dataset, label="Class")

# Initialize the model
RF_model = tfdf.keras.RandomForestModel()

# Balance class weights
class_weight = {0: (1/2204)*(2546/2.0), 1: (1/342)*(2546/2.0)}
RF_model.fit(tfdf_train, class_weight=class_weight)

# Evaluate model
# Metrics are available here: https://www.tensorflow.org/api_docs/python/tf/keras/metrics
RF_model.compile(metrics=["accuracy", "TruePositives", "TrueNegatives", "FalsePositives", "FalseNegatives",
                          tf.keras.metrics.AUC(name='roc', curve='ROC'), tf.keras.metrics.AUC(name='pr', curve='PR')])
evaluation = RF_model.evaluate(tfdf_test, return_dict=True)

# This one prints lower than average metrics (sensitivity and PR AUC)
# These *might* be some reasons why results are bad:
# Features are still incomplete
# We haven't addressed class imbalance issue: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
# TFDF library's implementation is not good (not likely the case)
for name, value in evaluation.items():
    print(f"{name}: {value:.4f}")

# Save model
RF_model.save("models/RF_model")
