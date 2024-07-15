import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

data = pd.read_csv("data.csv")


# * Simple linear regression With Gradient Descent
def loss_function(m, b, points: DataFrame):
    total_loss = 0
    for i in range(len(points)):
        x = points.iloc[i]["x"]
        y = points.iloc[i][" y"]
        total_loss += (y - (m * x + b)) ^ 2
    return total_loss / len(points)


def gradient_descent(m_now, b_now, points: DataFrame, lr):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i]["x"]
        y = points.iloc[i][" y"]

        m_gradient += -(2 / n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2 / n) * (y - (m_now * x + b_now))

    m = m_now - lr * m_gradient
    b = b_now - lr * b_gradient

    return m, b


m_ = 0
b_ = 0
lr_ = 0.001
epochs = 1000

for i in range(epochs):
    m_, b_ = gradient_descent(m_, b_, data, lr_)

print("gradient m:", m_, "gradient b:", b_)

plt.scatter(data["x"], data[" y"])
plt.plot(list(range(0, 30)), [m_ * x + b_ for x in range(0, 30)], color="red")
plt.show()
