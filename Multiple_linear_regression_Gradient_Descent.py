from typing import Any

import numpy as np
from sklearn.datasets import make_regression

X, y, coff_ = make_regression(
    n_samples=100, n_features=10, random_state=1, coef=True, bias=7.54, noise=5.0
)


# ? Class without using Matrix algebra Slow implementation
class MultipleLinearRegressionGD:
    def __init__(self, epochs: int, lr: float) -> None:
        self.coefficients = Any
        self.intercept = Any
        self.epochs = epochs
        self.lr = lr

    def fit(self, X: np.ndarray[Any, Any], y: np.ndarray):
        self.X_train = np.insert(X, 0, 1, axis=1)
        self.y_train = y
        N = self.X_train.shape[0]
        betas = np.array([1] * self.X_train.shape[1])
        betas[0] = 0

        for _ in range(1000):
            gradients = []

            for col in range(self.X_train.shape[1]):
                grad = 0
                for row in range(self.X_train.shape[0]):
                    grad += (
                        -(2 / N)
                        * (self.y_train[row] - sum(self.X_train[row] * betas))
                        * self.X_train[row][col]
                    )
                gradients.append(grad)

            betas = betas - (self.lr * np.array(gradients))
        self.intercept = betas[0]
        self.coefficients = betas[1:]

        return self.coefficients, self.intercept

    def predict(self, X):
        return sum(X * self.coefficients) + self.intercept


# ? MLR With matrix algebra
class MLRGD:
    def __init__(self, epochs, lr):
        self.coefficients: Any
        self.epochs = epochs
        self.lr = lr

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.coefficients = np.ones(X.shape[1])
        self.coefficients[0] = 0

        for _ in range(self.epochs):
            yHat = np.dot(X, self.coefficients)
            gradient = -(2 / X.shape[0]) * (np.dot((y - yHat), X))
            self.coefficients = self.coefficients - self.lr * gradient

        return self.coefficients[1:], self.coefficients[0]

    def predict(self, x):
        return np.dot(x, self.coefficients[1:]) + self.coefficients[0]


lr = MLRGD(epochs=1000, lr=0.01)
b0 = lr.fit(X, y)

yHat = lr.predict(X[55])

print(f"Actual y: {y[55]}\nPredicted y: {yHat}")
# print(f"Actual codf: {coff_}\nPredicted coff: {b0}")

MLROLS = MultipleLinearRegressionGD(epochs=1000, lr=0.01)
b0 = MLROLS.fit(X, y)

yHat = MLROLS.predict(X[55])

print(f"Actual y: {y[55]}\nPredicted y: {yHat}")
# print(f"Actual codf: {coff_}\nPredicted coff: {b0}")
