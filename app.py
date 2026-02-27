import joblib
import numpy as np

# Load model
model = joblib.load("bus_model.pkl")

print("Bus Arrival Time Predictor 🚍")
print("Enter bus details:\n")

distance = float(input("Distance from stop (km): "))
traffic = int(input("Traffic level (1-5): "))
time_of_day = int(input("Time of day (hour in 24h format): "))

# Prepare input
input_data = np.array([[distance, traffic, time_of_day]])

prediction = model.predict(input_data)

print("\nEstimated Arrival Time:", round(prediction[0], 2), "minutes")