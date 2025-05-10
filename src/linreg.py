import pandas as pd
import numpy as np
from typing import Union

class LinearRegression:
    def __init__(self, epochs: int = 100, lr: float = 0.1) -> None:
        self._epochs = epochs
        self._lr = lr
        self._weights = None

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: Union[bool, int] = False):
        N = X.shape[0]
        X = np.concatenate([np.ones([N, 1]), X], axis=1)
        y = y.reshape(N, 1)

        self._weights = np.ones([X.shape[1], 1])

        for epoch in range(self._epochs):
            y_pred = X @ self._weights
            error = y - y_pred
            grad = (-1) * 2 * (X.T @ error) / N
            mse = 1 / N * np.sum(error**2)

            self._weights -= self._lr * grad

            if verbose:
                if epoch == 0:
                    print(f'start | loss: {mse}')
                elif epoch % verbose == 0:
                    print(f'{epoch} | loss: {mse}')

    def get_coef(self):
        return self._weights

    def __str__(self):
        representation = 'LinearRegression: '
        for key, value in self.__dict__.items():
            representation += f'{key.replace('_', '')}={value}' + ', '
        representation = representation[:-2]
        return representation
