import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import SGDRegressor, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
sgd_model = SGDRegressor(loss='squared_error', learning_rate='constant', eta0=0.01, max_iter=1, warm_start=True, penalty=None, random_state=42)

losses = []
for epoch in range(100):
    sgd_model.fit(X_train, y_train)
    y_pred = sgd_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    losses.append(mse)

print("\n SGDRegressor")
print("Coefficients:", sgd_model.coef_)
print("Intercept:", sgd_model.intercept_)
print("Final MSE:", losses[-1])
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
print("\n Ridge Regression")
print("Coefficients:", ridge_model.coef_)
print("Intercept:", ridge_model.intercept_)
print("MSE:", mse_ridge)
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
print("\n Lasso Regression")
print("Coefficients:", lasso_model.coef_)
print("Intercept:", lasso_model.intercept_)
print("MSE:", mse_lasso)
elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_model.fit(X_train, y_train)
y_pred_elastic = elastic_model.predict(X_test)
mse_elastic = mean_squared_error(y_test, y_pred_elastic)
print("\n ElasticNet Regression")
print("Coefficients:", elastic_model.coef_)
print("Intercept:", elastic_model.intercept_)
print("MSE:", mse_elastic)
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(range(1, 101), losses, color='red')
plt.title("SGDRegressor MSE over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(True)
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_ridge, color='blue', label="Ridge")
plt.scatter(y_test, y_pred_lasso, color='purple', label="Lasso")
plt.scatter(y_test, y_pred_elastic, color='green', label="ElasticNet", alpha=0.7)
plt.scatter(y_test, y_pred, color='red', label="SGD", alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.legend()
plt.title("Predicted vs Actual")
plt.subplot(1, 3, 3)
mse_values = [losses[-1], mse_ridge, mse_lasso, mse_elastic]
models = ["SGD", "Ridge", "Lasso", "ElasticNet"]
colors = ['red', 'blue', 'purple', 'green']
plt.bar(models, mse_values, color=colors)
plt.title("MSE Comparison")
plt.ylabel("MSE")
plt.tight_layout()
plt.show()
