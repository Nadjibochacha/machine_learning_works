import numpy as np
import matplotlib.pyplot as plt

# Cost Function
def Cost(X, y, theta):
    m = len(y)
    predictions = X @ theta
    errors = predictions - y
    cost = (1 / (2 * m)) * np.sum(errors ** 2)
    return cost

# Gradient Descent
def Gradient(X, y, theta, alpha, num_iters):
    m = len(y)
    cost_history = []

    for _ in range(num_iters):
        predictions = X @ theta
        errors = predictions - y
        gradient = (1 / m) * (X.T @ errors)
        theta = theta - alpha * gradient
        cost = Cost(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history

# Feature Normalization
def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

# One Variable Linear Regression
X1 = np.array([2104, 1600, 2400, 1461, 3000])
y1 = np.array([400, 330, 369, 232, 540])
m1 = len(y1)
X1 = X1.reshape((m1, 1))
X1 = np.hstack([np.ones((m1, 1)), X1])
y1 = y1.reshape((m1, 1))
theta1 = np.zeros((2, 1))
alpha1 = 0.0000001
iterations1 = 1000

theta1, cost_history1 = Gradient(X1, y1, theta1, alpha1, iterations1)
print("Trained parameters (ONE VAR):", theta1.ravel())
print("Final cost (ONE VAR):", cost_history1[-1])

# Plotting the regression line
plt.figure(figsize=(10, 4))

# Subplot 1: Data + Regression line
plt.subplot(1, 2, 1)
plt.scatter(X1[:, 1], y1, color='pink', label='Training data')
plt.plot(X1[:, 1], X1 @ theta1, color='red', label='Linear regression')
plt.xlabel('House Size')
plt.ylabel('Price')
plt.title('One Variable Linear Regression')
plt.legend()

# Subplot 2: Cost vs Iterations
plt.subplot(1, 2, 2)
plt.plot(range(iterations1), cost_history1, color='purple')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Convergence (One Variable)')

plt.tight_layout()
plt.show()

# Multiple Variable Linear Regression
X2 = np.array([
    [2104, 5, 1, 45],
    [1600, 3, 2, 40],
    [2400, 3, 2, 30],
    [1461, 2, 2, 34],
    [3000, 6, 1, 36]
])
y2 = np.array([400, 330, 369, 232, 540]).reshape(-1, 1)

X2, mu, sigma = feature_normalize(X2)
m2, n2 = X2.shape
X2 = np.hstack([np.ones((m2, 1)), X2])
theta2 = np.zeros((n2 + 1, 1))
alpha2 = 0.01
iterations2 = 2000

theta2, cost_history2 = Gradient(X2, y2, theta2, alpha2, iterations2)
print("Trained parameters (MULTI VAR):", theta2.ravel())
print("Final cost (MULTI VAR):", cost_history2[-1])

# Plot cost convergence for multiple variables
plt.figure(figsize=(6, 4))
plt.plot(range(iterations2), cost_history2, color='blue')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Convergence (Multiple Variables)')
plt.grid(True)
plt.show()
