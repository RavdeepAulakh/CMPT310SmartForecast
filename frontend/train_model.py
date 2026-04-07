import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Feature columns (in order, matching the data)
FEATURE_COLS = ['hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores']
MODEL_PATH = 'models/best_model.pkl'
SCALER_PATH = 'models/scaler.pkl'

print("Loading training data...")
df = pd.read_csv('../data/exams.csv')

# Extract features and target
X = df.drop(columns=["student_id", "exam_score"])
y = df["exam_score"]

print(f"Training on {len(X)} samples with features: {list(X.columns)}")

# Preprocess
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Linear Regression (simpler and often more accurate)
print("Training Linear Regression model...")
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_scaled, y)

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score
y_pred = model.predict(X_scaled)
mse = mean_squared_error(y, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y, y_pred)

print(f"\nModel Performance:")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Save model and scaler
print("\nSaving model...")
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"Model saved to {MODEL_PATH}")
print(f"Scaler saved to {SCALER_PATH}")
print("\nDone! You can now run the frontend server.")
