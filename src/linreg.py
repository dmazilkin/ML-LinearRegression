import pandas as pd
import numpy as np
import random
from typing import Union, List, Tuple, Dict, Callable


class LinearRegression:

    def __init__(self, epochs: int = 100, lr: Union[float, Callable] = 0.1, metrics: List[str] = None, reg: str = None, l1_coef: float = 0.0, l2_coef: float = 0.0, sgd_sample: Union[int, float] = None, random_state: int = 42) -> None:
        # initialize basic Linear Regression parameters
        self._epochs = epochs
        self._lr: Union[float, Callable] = lr
        self._weights = None

        # initialize metrics parameters
        self._metrics = [metric.lower() for metric in metrics] if metrics is not None else None
        self._score: Union[Dict[float], None] = dict() if metrics is not None else None
        self._best_score: Union[Dict[float], None] = dict() if metrics is not None else None

        # initialize regularization parameters
        self._reg: Union[str, None] = reg
        self._l1_coef: float = l1_coef
        self._l2_coef: float = l2_coef

        # initialize SGD parameters
        self._sgd_sample = sgd_sample
        self._random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: Union[bool, int] = False) -> Tuple[List[float], List[List[float]]]:
        # pandas to numpy
        X = X.to_numpy()
        y = y.to_numpy()

        # preprocess data: add bias to features, reshape target vector
        N = X.shape[0]
        X = np.concatenate([np.ones([N, 1]), X], axis=1)
        y = y.reshape(N, 1)

        # initialize weights as ones vector
        self._weights = np.ones([X.shape[1], 1])
        weights_history = np.array(self._weights)

        # set batch size as batch gradient descent
        batch_size = X.shape[0]

        # set seed and mini-batch size for sgd
        if self._sgd_sample is not None:
            random.seed(self._random_state)

            if isinstance(self._sgd_sample, int):
                batch_size = self._sgd_sample

            if isinstance(self._sgd_sample, float):
                batch_size = round(X.shape[0] * self._sgd_sample)

        loss_history = []

        # start training
        for epoch in range(self._epochs):
            X_indx = random.sample(range(X.shape[0]), N)
            batch_current = 0

            # start sgd for one epoch
            while batch_current < N:
                # calculate prediction
                y_pred = X @ self._weights

                # calculate error
                error = y - y_pred

                # calculate regularization if set
                reg = 0
                reg_grad = 0

                if self._reg is not None:
                    reg, reg_grad = self._calc_regularization()

                # calculate loss and gradient for MSE
                loss = 1 / N * np.sum(error ** 2) + np.sum(reg)
                loss_history.append(loss)

                if epoch > 0:
                    weights_history = np.concatenate([weights_history, self._weights], axis=1)

                batch_current_size = batch_size if batch_current + batch_size <= N else N % batch_size
                batch_indx = X_indx[batch_current:batch_current+batch_current_size]
                X_batch = X[batch_indx]
                grad = (-1) * 2 * (X_batch.T @ error[batch_indx]) / batch_current_size + reg_grad

                # update weights with gradient descent
                lr = self._lr if not callable(self._lr) else self._lr(epoch+1)
                self._weights -= lr * grad

                # increase mini-batch index
                batch_current += batch_current_size

            # update metrics values if metrics are set
            if self._metrics is not None:
                self._update_metric(X, y)

            # print epoch loss and metrics if verbose is set
            if verbose and (epoch % verbose == 0):
                self._verbose(X, y, loss, epoch)

            if self._metrics is not None:
                self._update_metric(X, y)

        return loss_history, weights_history

    def _calc_regularization(self):
        reg = 0
        reg_grad = 0

        if self._reg == 'l1':
            reg = self._l1_coef * np.abs(self._weights)
            reg_grad = self._l1_coef * np.sign(self._weights)

        if self._reg == 'l2':
            reg = self._l2_coef * self._weights**2
            reg_grad = self._l2_coef * 2 * self._weights

        if self._reg == 'elasticnet':
            reg = self._l1_coef * np.abs(self._weights) + self._l2_coef * self._weights**2
            reg_grad = self._l1_coef * np.sign(self._weights) + self._l2_coef * 2 * self._weights

        return reg, reg_grad

    def _update_metric(self, X, y):
        for metric in self._metrics:
            metric_value = self._get_metric_value(X, y, metric)
            self._score[metric] = metric_value

    def _get_metric_value(self, X, y, metric):
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

    def _verbose(self, X: np.array, y: np.array, loss: float, epoch: int) -> None:
        if epoch == 0:
            print(f'start | loss: {loss}', end='')
        else:
            print(f'{epoch} | loss: {loss}', end='')

        # calculate metric if defined and print it
        if self._metrics is not None:
            for metric_name, metric_value in self._score.items():
                print(f' | {metric_name}: {metric_value}', end='')

        # add new line
        print()

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
        return np.mean(self._weights)

    def __str__(self):
        representation = 'LinearRegression: '

        for key, value in self.__dict__.items():
            representation += f'{key.replace('_', '')}={value}' + ', '
        representation = representation[:-2]

        return representation
