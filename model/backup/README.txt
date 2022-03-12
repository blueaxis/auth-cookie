This directory contains the previously trained models.

Naming scheme:
model_FFFF_S; where 0.FFFF is the F2-score and S is the seed or the random_state

Current best random_state: 12
Specificity: 0.7995444191343963
Sensitivity/Recall: 0.9014084507042254
Precision: 0.42105263157894735
F2-Score: 0.7339449541284403
Confusion Matrix:
tp: 64 fn: 7
fp: 88 tn: 351
PR-AUC: 0.39326589049259436

The best-performing model is converted into a Tensorflow SavedModel and is stored in model_tfdf.