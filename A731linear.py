import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5],
              [5, 6], [6, 7], [7, 8], [8, 9]]) 
y = np.array([2, 4, 5, 4, 6, 7, 8, 9])  
model = LinearRegression()
model.fit(X, y)
new_data = np.array([[9, 10], [10, 11]])
predictions = model.predict(new_data)
print("Code by A-762 Yashika")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
plt.scatter(X[:, 0], y, color='blue', label='Actual Data Points')
y_pred = model.predict(X)
plt.plot(X[:, 0], y_pred, color='red', label='Regression Line')
plt.scatter(new_data[:, 0], predictions, color='green', marker='x', s=100, label='New Data Predictions')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()
