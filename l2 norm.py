
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# 1. Load Diabetes Dataset
data = load_diabetes()
print(data.DESCR)

X = data.data
y = data.target

# 2. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=45
)

# 3. Linear Regression
L = LinearRegression()
L.fit(X_train, y_train)

print("\nLinear Regression")
print("Coefficients:", L.coef_)
print("Intercept:", L.intercept_)

y_pred = L.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# 4. Ridge Regression
R = Ridge(alpha=100000)
R.fit(X_train, y_train)

print("\nRidge Regression")
print("Coefficients:", R.coef_)
print("Intercept:", R.intercept_)

y_pred1 = R.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred1))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred1)))

# 5. Generate Polynomial Dataset
m = 100

x1 = 5 * np.random.rand(m, 1) - 2
x2 = 0.7 * x1**2 - 2*x1 + 3 + np.random.randn(m, 1)

plt.scatter(x1, x2)
plt.title("Generated Dataset")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

# 6. Ridge Regression with Polynomial Features
def get_preds_ridge(x1, x2, alpha):

    model = Pipeline([
        ('poly_feats', PolynomialFeatures(degree=16)),
        ('ridge', Ridge(alpha=alpha))
    ])

    model.fit(x1, x2)

    return model.predict(x1)

# 7. Different Alpha Values
alphas = [0, 20, 200]
colors = ['r', 'g', 'b']

plt.figure(figsize=(10, 6))

plt.scatter(x1, x2, label="Data Points")

# Sort X values for smooth plotting
x_sorted = np.sort(x1[:, 0])
x_sorted_2d = x_sorted.reshape(-1, 1)

for alpha, c in zip(alphas, colors):

    preds = get_preds_ridge(x1, x2, alpha)

    plt.plot(
        x_sorted,
        preds[np.argsort(x1[:, 0])].ravel(),
        c,
        label=f"Alpha: {alpha}"
    )

plt.title("Ridge Regression with Different Alpha Values")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.show()
