import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.linreg import LinearRegression

def main():

    X_data = np.random.rand(100, 1) * 10
    X_real = np.arange(X_data.min(), X_data.max(), step=0.1)
    X_real = X_real.reshape(X_real.shape[0], 1)

    y_data = 3 * X_data + 5 + np.random.randn(100, 1) * 2
    y_real = 3 * X_real + 5

    model = LinearRegression(epochs=1000, lr=0.01, metrics=['mse', 'rmse', 'mape', 'r2'])
    model.fit(pd.DataFrame(X_data), pd.Series(y_data.reshape(-1)), verbose=10)
    prediction = model.predict(pd.DataFrame(X_real)).to_numpy()

    plt.plot(X_real, y_real, c='g', marker='_')
    plt.scatter(X_data, y_data, c='b', marker='*')
    plt.plot(X_real, prediction, c='r', marker='+')
    plt.show()

    print(model.get_best_score())

if __name__ == '__main__':
    main()