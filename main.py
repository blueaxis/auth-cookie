# TFDF Documentation: https://www.tensorflow.org/decision_forests/

# Note regarding tfdf module
# It cannot be installed using Windows pip
# Either use Linux or WSL
import tensorflow_decision_forests as tfdf
import tensorflow as tf
import pandas as pd

from collections import Counter
import numpy as np

# Read dataset from CSV file
col_names = ["Website", "ID", "Name", "Value", "Domain", "Path", "Secure", "Expiry", "HTTP-Only", "JavaScript", "Class"]
dataset = pd.read_csv("data/cookies.csv", names=col_names)

# TODO: Implement all helper functions
# Helper functions for feature conversion
# *******************************************


COOKIE_ALPHABET = "abdefghijklmnqrstuvxyzABDEFGHIJKLMNQRSTUVXYZ0123456789!#$%&'()*+-./:<>?@[]^_`{|}~"


def index_of_coincidence(c_value):
    c_value = str(c_value)
    ctr = Counter(c_value)
    n = len(c_value)
    numerator = 0
    for c in COOKIE_ALPHABET:
        numerator += ctr[c] * (ctr[c] - 1)
    try:
        return numerator / (n * (n - 1))
    except ZeroDivisionError:
        return 1


def shannon_entropy(c_value, base=2):
    c_value = str(c_value)
    ctr = Counter(c_value)
    n = len(c_value)
    entropy = 0
    for _, c in ctr.items():
        entropy += (c / n) * np.log(c / n) / np.log(base)
    return -entropy


# *******************************************


# Feature conversion
# *******************************************

# Naming Scheme
STANDARD_NAMES = ["JSESSIONID", "ASPSESSIONIDXXXXXXXX", "ASP.NET_SessionId", "PHPSESSION", "wiki18_session",
                  "WebLogicSession", "BIGipServerxxx_xxx_xxx_PROTO", "SERVERID", "SaneID", "ssuid", "vgnvisitor",
                  "SESSION_ID", "NSES40Session", "iPlanetUserId", "gx_session_id_", "JROUTE", "RMID", "JSESSIONID",
                  "Apache", "CFID", "CFTOKEN", "CFGLOBALS", "RoxenUserID", "JServSessionIdroot", "sesessionid",
                  "PD-S-SESSION-ID", "PD_STATEFUL", "WEBTRENDS_ID", "__utmX", "sc_id", "s_sq", "s_sess", "s_vi_XXXXXX",
                  "MintUnique", "MintXXXX", "SS_X_CSINTERSESSIONID", "CSINTERSESSIONID", "_sn", "BCSI-CSCXXXXXXX",
                  "Ltpatoken", "Ltpatoken2", "LtpatokenExpiry", "LtpatokenUsername", "DomAuthSessID", "connect.sid"]
dataset["Scheme"] = dataset["Name"].map(lambda n: 1 if (n in STANDARD_NAMES) else 0)

# Expiration
dataset["Expiry"] = dataset["Expiry"].map(lambda e: e - 1370000000)

# Index of Coincidence
dataset["IC"] = np.asarray(dataset["Value"].map(index_of_coincidence)).astype(np.float32)

# Shannon Entropy
dataset["Entropy"] = np.asarray(dataset["Value"].map(shannon_entropy)).astype(np.float32)

# Length
dataset["Length"] = dataset["Value"].map(lambda v: len(str(v)))

# *******************************************


# Remove unused columns
dataset = dataset.drop(
    columns=["Website", "ID", "Name", "Value", "Domain", "Path", "Secure", "HTTP-Only", "JavaScript"])

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
RF_model.compile(metrics=["accuracy", "TruePositives", "TrueNegatives", "FalsePositives", "FalseNegatives", tf.keras.metrics.AUC(name='roc', curve='ROC'), tf.keras.metrics.AUC(name='pr', curve='PR')])
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

# TODO: Understand what the summary prints
# RF_model.summary()
