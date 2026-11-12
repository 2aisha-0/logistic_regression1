# Logistic Regression with different optimizers

This is my second ML assignment :)
I used logistic regression (binary classification) to predict a person's fitness level, and I wrote every optimizer by hand with numpy so I can see how each one works.

## Dataset
Body Performance Data from Kaggle (`bodyPerformance.csv`), 13,393 people.

- Features: age, gender, height_cm, weight_kg, body fat_%, diastolic, systolic, gripForce, sit and bend forward_cm, sit-ups counts, broad jump_cm
- Target: `class` (A, B, C, D) turned into binary:
  - A or B → **1** (high fitness)
  - C or D → **0** (lower fitness)

## Notebooks
| Notebook | Optimizer |
|---|---|
| 1_Gradient_Descent.ipynb | Gradient Descent (batch) |
| 2_Stochastic_Gradient_Descent.ipynb | Stochastic Gradient Descent |
| 3_Mini_Batch_Gradient_Descent.ipynb | Mini-Batch Gradient Descent |
| 4_RMSprop.ipynb | RMSprop |
| 5_Adam.ipynb | Adam |

## Steps (same in all notebooks)
1. load the data
2. change gender to 0 / 1 and make the binary target
3. split 80% train, 20% test, then scale
4. write sigmoid, binary cross-entropy loss and the gradient
5. train with the optimizer
6. plot the loss, show the confusion matrix and compute Accuracy, Precision, Recall, F1 and Loss on the test set
7. check our hand-written metrics against scikit-learn

## Results (test set)
| Optimizer | Loss | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| GD | 0.391 | 0.824 | 0.811 | 0.845 | 0.828 |
| SGD | 0.396 | 0.823 | 0.821 | 0.825 | 0.823 |
| Mini-Batch GD | 0.392 | 0.823 | 0.811 | 0.843 | 0.827 |
| RMSprop | 0.392 | 0.821 | 0.813 | 0.834 | 0.823 |
| Adam | 0.392 | 0.823 | 0.808 | 0.846 | 0.827 |

All five reached almost the same result. The logistic regression loss is convex (it has only one minimum), so every optimizer ends up in the same place. The difference is in how they get there: GD needed 300 epochs, SGD is the noisiest, and Mini-Batch, RMSprop and Adam got there in 50 epochs.

## Run it
Open any notebook on Kaggle, add the Body Performance Data dataset as input, then Run All.

Libraries: numpy, pandas, matplotlib, scikit-learn
