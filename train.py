# Train and optimize random forest hyperparameters
import pandas as pd
import keras_tuner as kt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics
import pickle

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

def build_model(hp):
    model = RandomForestClassifier(
        n_estimators= hp.Int("n_estimators", 100, 300, step= 10),
        criterion= hp.Choice("criterion", ["gini", "entropy"]),
        max_depth= hp.Int("max_depth", 6, 9, step= 1),
        min_samples_split= hp.Int("min_samples_split", 20, 40, step= 1),
        max_features= hp.Choice("max_features", ["auto", "log2"]),
        class_weight= {0: (2546 / (2204 * 2.0)), 1: (2546 / (342 * 2.0))},
        max_leaf_nodes= hp.Int("max_leaf_nodes", 70, 160, step= 5),
        oob_score= "True",
    )
    return model

# Initialize the Bayesian Search Tuner that maximizes the F2-score
tuner = kt.tuners.SklearnTuner(
    oracle=kt.oracles.BayesianOptimizationOracle(
        objective=kt.Objective("score", "max"),
        max_trials=100
		),
    hypermodel=build_model,
    scoring=metrics.make_scorer(metrics.fbeta_score, beta=2),
    cv=StratifiedKFold(10),
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

#save model
filename = 'model.sav'
pickle.dump(best_model, open(filename, 'wb'))