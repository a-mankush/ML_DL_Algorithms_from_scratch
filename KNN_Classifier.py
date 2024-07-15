from collections import Counter

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNNClassifier:
    def __init__(self, k: int = 3) -> None:
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        distances = [euclidean_distance(x, X_train) for X_train in self.X_train]
        sorted_indexes = sorted(range(len(distances)), key=lambda k: distances[k])
        top_k_sorted_indexes = sorted_indexes[: self.k]
        y_values = [self.y_train[index] for index in top_k_sorted_indexes]
        counter = Counter(y_values)
        return counter.most_common(1)[0][0]


X, y = make_classification(
    n_samples=100, n_features=5, n_classes=3, n_clusters_per_class=2, n_informative=3
)

knn = KNNClassifier()
knn.fit(X, y)
preds = knn.predict(X)
print(preds[:5])
print(y[:5])

print(accuracy_score(y, preds))
