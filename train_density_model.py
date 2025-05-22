from sklearn.linear_model import LogisticRegression
import joblib

# Simulated data: [total_vehicles, heavy_vehicles] → density (0=Low, 1=Medium, 2=High)
X = [
    [2, 0], [4, 0], [5, 1],      # Low
    [8, 2], [12, 1], [10, 3],    # Medium
    [16, 5], [18, 3], [22, 6]    # High
]
y = [0, 0, 0, 1, 1, 1, 2, 2, 2]

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "traffic_density_model.pkl")
print("✅ Model trained and saved!")
