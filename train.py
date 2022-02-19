# Train and optimize random forest hyperparameters
# Suggestions:
# 1. Use more hyperparameters from the documentation.
# 2. Increase hyperparameter ranges for numerical values
# i.e. use more than one decimal place, lower min, higher max, etc.
# 3. Use multiple scoring metrics if possible. Also, there might
# be a better metric to optimize the model


import pandas as pd
import keras_tuner as kt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics


# Read training features
ds = pd.read_csv("data/features.csv")

# Normalize dataset
# Minmax normalization is commented
normalized_ds = (ds - ds.mean()) / ds.std()  # (ds-ds.min())/(ds.max()-ds.min())
normalized_ds["Class"] = ds["Class"]
normalized_ds["Length"] = ds["Length"]
normalized_ds["TFIDF_H"] = ds["TFIDF_H"]
normalized_ds["TFIDF_S"] = ds["TFIDF_S"]
normalized_ds["TFIDF_J"] = ds["TFIDF_J"]
normalized_ds["Z_Length"] = ds["Z_Length"]

# Split dataset
y = normalized_ds[["Class"]]
X = normalized_ds.drop("Class", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Note: Anything beyond this point are only experiments
# Though you may use them as reference
# Most of the code is copied from this source:
# https://keras.io/api/keras_tuner/tuners/sklearn/


# Initialize model with hyperparameter values within search space
# Look for available hyperparameters and corresponding ranges here:
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
def build_model(hp):
    # Int, Float, Choice usage here:
    # https://keras.io/api/keras_tuner/hyperparameters/
    # Note: This will only optimize the hyperparameters listed in the constructor
    # Add more hyperparameters with larger ranges
    model = RandomForestClassifier(
        n_estimators=hp.Int("n_estimators", 100, 500, step=10),
        criterion=hp.Choice("criterion", ["gini", "entropy"]),
        max_depth=hp.Int("max_depth", 4, 32, step=1),
        min_samples_split=hp.Int("min_samples_split", 2, 20, step=1),
        min_samples_leaf=hp.Int("min_samples_leaf", 1, 5, step=1),
        max_features=hp.Choice("max_features", ["auto", "sqrt", "log2"]),
        class_weight={0: (2546 / (2204 * 2.0)), 1: (2546 / (342 * 2.0))}
        # Try to implement class_weight as a hyperparameter
        # The reason why this does not work is dictionaries
        # cannot be passed into a Choice object
        # Maybe you can find a workaround for this
        # class_weight=hp.Choice("class_weight",
        #                       [{0: 1, 1: 1}, {0: 1, 1: 2}, {0: 1, 1: 4},
        #                       {0: 1, 1: 8}, {0: 1, 1: 16}, {0: 1, 1: 32}])
    )
    return model


# Initialize the Bayesian Search Tuner that maximizes the F2-score
tuner = kt.tuners.SklearnTuner(

    # Do not change the oracle objective
    # The number of trials is the number of hyperparameter combinations that will be tested
    # We can use the Deep Learning Workstation if our machines cannot handle long searches
    oracle=kt.oracles.BayesianOptimizationOracle(
        objective=kt.Objective("score", "max"),
        max_trials=100),
    hypermodel=build_model,

    # Random forest model is optimized based on the scoring variable
    # Available metrics: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    # No idea if multiple scoring metrics is possible
    scoring=metrics.make_scorer(metrics.fbeta_score, beta=2),
    cv=StratifiedKFold(10),

    # Trials are saved in this directory
    # You have to delete this directory if you want to run train.py again
    # i.e. if you want to find better hyperparameters
    # No idea if this can be done automatically after each run
    directory='./model/',
    project_name='random_forest'
    )

# Search for the best hyperparameter values
tuner.search(X_train, y_train.values.ravel())
best_hp = tuner.get_best_hyperparameters()[0]

# Build and train model
best_model = tuner.hypermodel.build(best_hp)
best_model.fit(X_train, y_train.values.ravel())

# Evaluate using test set
y_pred = best_model.predict(X_test)
print("PR-AUC:", metrics.average_precision_score(y_test, y_pred))
tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
print("Specificity:", tn / (tn + fp))
print("Sensitivity/Recall:", metrics.recall_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("F2-Score:", metrics.fbeta_score(y_test, y_pred, beta=2))
print("Confusion Matrix:")
print("tp:", tp, "fn:", fn)
print("fp:", fp, "tn:", tn)

# These are the best hyperparameters from my testing
# Hyperparameter        |Best Value So Far
# n_estimators          | 150
# criterion             | entropy
# max_depth             | 7
# min_samples_split     | 16
# min_samples_leaf      | 2
# max_features          | auto
# And the corresponding evaluation metrics
# PR-AUC: 0.34888310401918826
# Specificity: 0.8036117381489842
# Sensitivity/Recall: 0.835820895522388
# Precision: 0.3916083916083916
# F2-Score: 0.6812652068126521
# Confusion Matrix:
# tp: 56 fn: 11
# fp: 87 tn: 356