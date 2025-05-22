import joblib
from sklearn.linear_model import LinearRegression
import numpy as np

# Example training data: [vehicle_count, heavy_vehicle_count] → green light time (in seconds)
X = np.array([
    [5, 1],
    [10, 2],
    [15, 3],
    [20, 4],
    [25, 5],
    [30, 6]
])

# Corresponding green light times in seconds
y = np.array([15, 25, 35, 45, 55, 65])

# Initialize and train the model
green_light_model = LinearRegression()
green_light_model.fit(X, y)

# Save the trained model to a file
joblib.dump(green_light_model, "green_light_time_model.pkl")
print("Green light time model saved as 'green_light_time_model.pkl'")
