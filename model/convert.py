# TFDF Documentation: https://www.tensorflow.org/decision_forests/
import pandas as pd
import tensorflow_decision_forests as tfdf
import tensorflow as tf

# Read dataset from CSV file
ds = pd.read_csv("data/features.csv")

# Normalize
normalized_ds = (ds - ds.mean()) / ds.std()  # (ds-ds.min())/(ds.max()-ds.min())
normalized_ds["Class"] = ds["Class"]
normalized_ds["Length"] = ds["Length"]
normalized_ds["TFIDF_H"] = ds["TFIDF_H"]
normalized_ds["TFIDF_S"] = ds["TFIDF_S"]
normalized_ds["TFIDF_J"] = ds["TFIDF_J"]
normalized_ds["Z_Length"] = ds["Z_Length"]

# Split dataset into training and test set (8:2) ratio
train_dataset = normalized_ds.sample(frac=0.8, random_state=12)
test_dataset = normalized_ds.drop(train_dataset.index)

# No idea if this is needed
tfdf_train = tfdf.keras.pd_dataframe_to_tf_dataset(train_dataset, label="Class")
tfdf_test = tfdf.keras.pd_dataframe_to_tf_dataset(test_dataset, label="Class")

# Initialize the model
# Hyperparamaters arrangement should be
# the same with the sklearn model to avoid confusion
RF_model = tfdf.keras.RandomForestModel(
    num_trees=500,                  # n_estimators
    # criterion="entropy",
    max_depth=6,                    # max_depth
    min_examples=20,                # min_samples_split
    num_candidate_attributes=0,     # max_features
    max_num_nodes=160,              # max_leaf_nodes
    bootstrap_size_ratio=1.0,       # max_samples
    # class_weight="balanced",
    # oob_score=True,
    growing_strategy="BEST_FIRST_GLOBAL"
)
# Balance class weights
class_weight = {0: (2546 / (2204 * 2.0)), 1: (2546 / (342 * 2.0))}
RF_model.fit(tfdf_train, class_weight=class_weight)

# Evaluate model
# Available metrics: https://www.tensorflow.org/api_docs/python/tf/keras/metrics
RF_model.compile(metrics=["Precision", "Recall", "TruePositives", "TrueNegatives",
                          "FalsePositives", "FalseNegatives",
                          tf.keras.metrics.AUC(name='pr', curve='PR')])
evaluation = RF_model.evaluate(tfdf_test, return_dict=True)
for name, value in evaluation.items():
    print(f"{name}: {value:.4f}")

# Save model
RF_model.save("backup/model_tfdf")
