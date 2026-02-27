import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Simulated dataset
distance = np.array([5, 8, 10, 3, 7, 12, 6]).reshape(-1, 1)
traffic = np.array([2, 4, 5, 1, 3, 5, 2]).reshape(-1, 1)
time_of_day = np.array([9, 18, 20, 7, 14, 22, 10]).reshape(-1, 1)

# Combine features
X = np.hstack((distance, traffic, time_of_day))

# Target variable
arrival_time = np.array([12, 25, 35, 8, 20, 40, 15])

# Train model
model = LinearRegression()
model.fit(X, arrival_time)
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Save model
joblib.dump(model, "bus_model.pkl")

print("Model trained and saved.")