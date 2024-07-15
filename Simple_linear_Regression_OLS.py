# * Simple linear Regression OLS
import pandas as pd
from matplotlib import pyplot as plt


class SimpleLinearRegressionOLS:
    def __init__(self) -> None:
        self.b = None
        self.m = None

    def fit(self, x, y):

        numerator = 0
        denominator = 0

        for i in range(x.shape[0]):
            numerator += (x[i] - x.mean()) * (y[i] - y.mean())
            denominator += (x[i] - x.mean()) ** 2

        self.m = numerator / denominator
        self.b = y.mean() - (self.m * x.mean())

        return self.m, self.b

    def predict(self, x):
        return (self.m * x) + self.b


data = pd.read_csv("data.csv")

X = data["x"]
y = data[" y"]

SLR = SimpleLinearRegressionOLS()
m, b = SLR.fit(X, y)
print("m:", m)
print("b:", b)

yHat = SLR.predict(X[7])

plt.scatter(data["x"], data[" y"])
plt.plot(list(range(0, 22)), [m * x + b for x in range(0, 22)], color="red")
plt.show()
