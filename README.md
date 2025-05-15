# Linear Regression

This repository contains **Linear Regression** implementation from scratch.

## Features
This implementation contains training with batch and mini-batch gradient descent. To reproduce the results *random_state* parameter can be used.

Implemented methods of *regularization*:
  - L1 (Lasso),
  - L2 (Ridge),
  - ElasticNet,

Custom learning rate (float or callable per epoch) can be set during model initialization.

Implemented and supported metrics:
  - MSE,
  - RMSE,
  - MAE,
  - MAPE,
  - R² score,

Verbose training output can be set for every *n* epochs.

### Install Dependencies
```console
pip install requirements.txt
```

### Run simple Linear Regression example
```console
python main.py
```

### Run Linear Regression for House Pricing Prediction
The notebook *notebooks/kaggle_example.ipynb* demonstrates how to use the implemented from scratch Linear Regression on a real-world dataset.

1. First, you need to download dataset from Kaggle: [House Pricing Prediction](https://www.kaggle.com/code/dvidpterbihari/house-pricing-prediction-project).
2. Create *data/* directory in the root of the project, rename the downloaded file to "housing.csv" and put there.
3. Go to *notebooks/* directory and run *housing.ipynb* notebook.


