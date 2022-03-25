This directory contains the previously trained models.

Naming scheme:
model_FFFF_S; where 0.FFFF is the F2-score and S is the seed or the random_state

Current best random_state: 12
Specificity: 0.8246013667425968
Sensitivity/Recall: 0.9295774647887324
Precision: 0.46153846153846156
F2-Score: 0.7728337236533959
Confusion Matrix:
tp: 66 fn: 5
fp: 77 tn: 362
PR-AUC: 0.43883967454804246

Best hyperparameters:
n_estimators: 180
criterion: "entropy"
max_depth: 10
min_samples_split: 70
max_features: "auto"
max_leaf_nodes: 125
max_samples: 0.63
class_weight: "balanced"
oob_score: "True"