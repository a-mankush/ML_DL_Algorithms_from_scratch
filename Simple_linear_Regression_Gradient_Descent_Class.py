import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series


# * Simple linear regression With Gradient Descent
class SimpleLinearRegressionGD:
    def __init__(self) -> None:
        self.m: float = 0
        self.b: float = 0

    def fit(self, X: Series, y: Series) -> tuple[float, float]:
        n: int = len(X)  # Get the number of data points
        lr: float = 0.0001  # Set the learning rate

        for _ in range(
            10000
        ):  # Iterate for a number of iterations (10000 in this case)
            m_gradient: float = 0  # Initialize the gradient for the slope to 0
            b_gradient: float = 0  # Initialize the gradient for the intercept to 0

            for i in range(len(X)):  # For each data point
                # Calculate the gradient of the loss function with respect to the slope (m) and the intercept (b)
                m_gradient += -(2 / n) * X[i] * (y[i] - (self.m * X[i] + self.b))
                b_gradient += -(2 / n) * (y[i] - (self.m * X[i] + self.b))

            # Update the slope (m) and intercept (b) by subtracting the learning rate multiplied by the updated gradients
            self.m = self.m - (lr * m_gradient)
            self.b = self.b - (lr * b_gradient)

        return (
            self.m,
            self.b,
        )  # Return the updated slope (m) and intercept (b) as a tuple

    def predict(self, X):
        return (self.m * X) + self.b


data = pd.read_csv("data.csv")

X = data["x"]
y = data[" y"]

SLRGD = SimpleLinearRegressionGD()
m_value, b_value = SLRGD.fit(X, y)
print(m_value, b_value)

yHat = SLRGD.predict(X[7])

plt.scatter(data["x"], data[" y"])
plt.plot(list(range(0, 22)), [m_value * x + b_value for x in range(0, 22)], color="red")
plt.show()
