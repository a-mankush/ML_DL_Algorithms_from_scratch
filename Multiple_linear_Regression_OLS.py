import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_regression


class MultipleLinearRegressionOLS:
    def __init__(self):
        self.coefficient: list = []

    def fit(self, x, y):
        x = np.insert(x, 0, 1, axis=1)
        self.coefficient = np.linalg.inv(x.T.dot(x)).dot(x.T.dot(y))
        return self.coefficient

    def predict(self, x):
        return x.dot(self.coefficient[1:]) + self.coefficient[0]


X, y, coff_ = make_regression(
    n_samples=100, n_features=10, random_state=1, coef=True, bias=7.54, noise=5.0
)

MLROLS = MultipleLinearRegressionOLS()
b0 = MLROLS.fit(X, y)

yHat = MLROLS.predict(X[55])

print(f"Actual y: {y[55]}\nPredicted y: {yHat}")
print(f"Actual codf: {coff_}\nPredicted coff: {b0}")
