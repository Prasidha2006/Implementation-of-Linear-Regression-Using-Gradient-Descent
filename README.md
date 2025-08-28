# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Load dataset (R&D Spend → X, Profit → Y).

2. Scale X and Y.

3. Add bias term, initialize θ.

4. Define cost function and gradient descent update rule.

5. Run iterations to optimize θ.

6. Take user input (R&D Spend), scale it, and predict Profit.

7. Convert prediction back to original scale and plot regression line.

## Program:

```
# Program Developed by: PRASIDHA A - 212224230204

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("50_Startups.csv")

# Use R&D Spend as X and Profit as Y
X = data["R&D Spend"].values
y = data["Profit"].values
m = len(y)

# Feature Scaling
X_scaled = (X - np.mean(X)) / np.std(X)
y_scaled = (y - np.mean(y)) / np.std(y)

# Add bias term
X_scaled = np.c_[np.ones(m), X_scaled]
theta = np.zeros(2)

# Cost function
def compute_cost(X, y, theta):
    predictions = X.dot(theta)
    errors = predictions - y
    return (1/(2*m)) * np.dot(errors.T, errors)

# Gradient Descent
def gradient_descent(X, y, theta, alpha, iterations):
    cost_history = []
    for _ in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        theta = theta - (alpha/m) * (X.T.dot(errors))
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history

# Train model
alpha = 0.01   # Works well after scaling
iterations = 1000
theta, cost_history = gradient_descent(X_scaled, y_scaled, theta, alpha, iterations)

print("Optimized theta values (scaled):", theta)

# Prediction with user input
user_input = float(input("Enter R&D Spend value: "))

# Scale the input
user_input_scaled = (user_input - np.mean(X)) / np.std(X)

# Predict on scaled model
predicted_scaled = np.dot([1, user_input_scaled], theta)

# Convert back to original scale
predicted_profit = predicted_scaled * np.std(y) + np.mean(y)

print(f"Predicted Profit for R&D Spend {user_input} is: {predicted_profit:.2f}")

# Plot regression line
plt.scatter(X, y, label="Training data")
plt.plot(X, (X_scaled.dot(theta))*np.std(y)+np.mean(y), color="red", label="Regression line")
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.legend()
plt.show()

```

## Output:


<img width="864" height="615" alt="Screenshot 2025-08-28 144409" src="https://github.com/user-attachments/assets/0a062add-f159-4629-a7da-218c41ff87ba" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
