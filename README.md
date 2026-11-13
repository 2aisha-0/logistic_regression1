# Logistic Regression with 5 optimizers

ML assignment 2

i used logistic regression (binary classification) to predict if a person has good fitness or not, and i wrote the 5 optimizers myself with numpy

## Dataset
Body Performance Data from kaggle (13393 rows)

the class column has A, B, C, D so i changed it to binary:
- A and B = 1 (good fitness)
- C and D = 0

## Notebooks
- 1_Gradient_Descent.ipynb
- 2_Stochastic_Gradient_Descent.ipynb
- 3_Mini_Batch_Gradient_Descent.ipynb
- 4_RMSprop.ipynb
- 5_Adam.ipynb

## Steps
1. load the data
2. change gender to 0/1 and make the target binary
3. train test split (80/20) and scaling
4. sigmoid, loss (binary cross entropy) and gradient functions
5. train with the optimizer
6. loss curve, accuracy, precision, recall, f1, loss and confusion matrix

## Results (test set)
| Optimizer | Accuracy | Precision | Recall | F1 | Loss |
|---|---|---|---|---|---|
| GD | 0.822 | 0.819 | 0.831 | 0.825 | 0.392 |
| SGD | 0.826 | 0.815 | 0.847 | 0.831 | 0.392 |
| Mini Batch | 0.823 | 0.820 | 0.830 | 0.825 | 0.392 |
| RMSprop | 0.825 | 0.821 | 0.835 | 0.828 | 0.392 |
| Adam | 0.821 | 0.819 | 0.827 | 0.823 | 0.393 |

all of them got around 82% accuracy. i think its because the logistic regression loss is convex so they all reach the same minimum, the difference is just how fast they get there (GD needed 300 epochs, the others 20-50)

## How to run
open the notebook on kaggle, add the Body Performance Data dataset and run all
