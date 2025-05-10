import pandas as pd
import numpy as np
from typing import Union, List

class LinearRegression:

    def __init__(self, epochs: int = 100, lr: float = 0.1, metrics: List[str] = None) -> None:
        self._epochs = epochs
        self._lr = lr
        self._weights = None
        self._metrics = [metric.lower() for metric in metrics] if metrics is not None else None
        self._best_score = dict() if metrics is not None else None

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: Union[bool, int] = False):
        # pandas to numpy
        X = X.to_numpy()
        y = y.to_numpy()
        # preprocess data: add bias to features, reshape target vector
        N = X.shape[0]
        X = np.concatenate([np.ones([N, 1]), X], axis=1)
        y = y.reshape(N, 1)
        # initialize weights as ones vector
        self._weights = np.ones([X.shape[1], 1])

        for epoch in range(self._epochs):
            # calculate prediction
            y_pred = X @ self._weights
            # calculate error
            error = y - y_pred
            # calculate mse and gradient for MSE
            mse = 1 / N * np.sum(error ** 2)
            grad = (-1) * 2 * (X.T @ error) / N
            # update weights with gradient descent
            self._weights -= self._lr * grad

            # print loss if verbose is set
            if verbose:
                if epoch == 0:
                    print(f'start | loss: {mse}', end='')
                elif epoch % verbose == 0:
                    print(f'{epoch} | loss: {mse}', end='')

            # calculate metric if defined and print if verbose is set
            if self._metrics is not None:
                for metric in self._metrics:
                    metric_value = self._get_metric(X, y, metric)
                    self._best_score[metric] = metric_value

                    if (self._metrics is not None) and (epoch % verbose == 0):
                        print(f' | {metric}: {metric_value}', end='')

            if verbose and epoch % verbose == 0:
                print()

    def _get_metric(self, X, y, metric):
        prediction = X @ self._weights
        if metric == 'mae':
            return np.mean(np.abs(y - prediction))
        if metric == 'mse':
            return np.mean((y - prediction)**2)
        if metric == 'rmse':
            return np.sqrt(np.mean((y - prediction)**2))
        if metric == 'mape':
            return 100 * np.mean(np.abs((y - prediction) / y))
        if metric == 'r2':
            return 1 - np.sum((y - X @ self._weights)**2) / np.sum((y - np.mean(y))**2)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        # preprocess data
        X = X.to_numpy()
        N = X.shape[0]
        X = np.concatenate([np.ones([N, 1]), X], axis=1)
        # calculate prediction
        prediction = X @ self._weights

        return pd.Series(prediction.reshape(-1))

    def get_best_score(self):
        return self._best_score

    def get_coef(self):
        return self._weights

    def __str__(self):
        representation = 'LinearRegression: '
        for key, value in self.__dict__.items():
            representation += f'{key.replace('_', '')}={value}' + ', '
        representation = representation[:-2]
        return representation
