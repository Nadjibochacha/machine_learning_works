import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Predict using the trained model
y_pred = model.predict(X)

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label="Data Points")  # Scatter plot of original data
plt.plot(X, y_pred, color='red', linewidth=2, label="Regression Line")  # Regression line
plt.xlabel("BMI-like Feature")
plt.ylabel("Target Variable")
plt.title("Linear Regression Plot")
plt.legend()
plt.show()
