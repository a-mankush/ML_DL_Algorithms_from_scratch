import numpy as np
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor


def euclidean_distance(x1, x2):
    return np.sqrt(sum((x1 - x2) ** 2))


class KNNRegressor:
    def __init__(self, k: int = 3):
        self.k: int = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        distance = [euclidean_distance(x, X_train) for X_train in self.X_train]
        top_k_index = np.argsort(distance)[: self.k]
        return sum([self.y_train[i] for i in top_k_index]) / len(top_k_index)


X, y, coff_ = make_regression(
    n_samples=100, n_features=10, random_state=1, coef=True, bias=7.54, noise=5.0
)

knn = KNNRegressor(k=5)
b0 = knn.fit(X, y)
yHat = knn.predict(X[:5])
print(f"Actual y: {y[:5]}\nPredicted y: {yHat}")

knn = KNeighborsRegressor(n_neighbors=5)
b0 = knn.fit(X, y)
yHat = knn.predict(X[:5])
print(f"Actual y: {y[:5]}\nPredicted y: {yHat}")
