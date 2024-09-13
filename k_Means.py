from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data[:, 2:4]
y = iris.target

y_0 = np.where(y == 0)
plt.scatter(X[y_0, 0], X[y_0, 1])
y_1 = np.where(y == 1)
plt.scatter(X[y_1, 0], X[y_1, 1])
y_2 = np.where(y == 2)
plt.scatter(X[y_2, 0], X[y_2, 1])

plt.show()

# k-Means

# Step 1
k = 3

# Step 2
random_index = np.random.choice(range(len(X)), k)
centroids = X[random_index]


def visualize_centroids(X, centroid):
    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(centroid[:, 0], centroid[:, 1], marker="*", s=200, c="#050505")
    plt.show()


visualize_centroids(X, centroids)


# Step 3
def dist(a, b):
    return np.linalg.norm(a - b, axis=1)


# Step 4
def assign_cluster(x, centroids):
    distances = dist(x, centroids)
    cluster = np.argmin(distances)
    return cluster


# Step 5
def update_centroids(X, centroids, clusters):
    for i in range(k):
        cluster_i = np.where(clusters == i)
        centroids[i] = np.mean(X[cluster_i], axis=0)


tol = 0.0001
max_iter = 100

iter_ = 0
centroids_diff = 100000
clusters = np.zeros(len(X))


while iter_ < max_iter and centroids_diff > tol:
    for i in range(len(X)):
        clusters[i] = assign_cluster(X[i], centroids)

    centroids_prev = deepcopy(centroids)
    update_centroids(X, centroids, clusters)
    iter_ += 1

    centroids_diff = np.linalg.norm(centroids - centroids_prev)

    print(f"Iteration: {iter_}/")
    print(f"Centorides: {centroids}")
    print(f"Centroids move: {centroids_diff:5.4f}")
    # visualize_centroids(X, centroids)


for i in range(k):
    cluster_i = np.where(clusters == i)
    plt.scatter(X[cluster_i, 0], X[cluster_i, 1])
plt.scatter(centroids[:, 0], centroids[:, 1], marker="*", s=200, c="#050505")
plt.show()
