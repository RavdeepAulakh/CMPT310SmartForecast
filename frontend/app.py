from flask import Flask, render_template, request, jsonify, redirect, url_for
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__, template_folder='../frontend/templates', static_folder='../frontend/static')

# Model and scaler paths
MODEL_PATH = 'models/best_model.pkl'
SCALER_PATH = 'models/scaler.pkl'

# Feature columns (in order)
FEATURE_COLS = ['hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores']


def load_model():
    """Load the best trained model."""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        print(f"Error: Model not found at {MODEL_PATH}")
        return None


def load_scaler():
    """Load the scaler used for preprocessing."""
    try:
        scaler = joblib.load(SCALER_PATH)
        return scaler
    except FileNotFoundError:
        print(f"Error: Scaler not found at {SCALER_PATH}")
        return None


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request from frontend."""
    try:
        data = request.get_json()

        # Extract input features (in the same order as training)
        features = {
            'hours_studied': float(data['hours_studied']),
            'sleep_hours': float(data['sleep_hours']),
            'attendance_percent': float(data['attendance_percent']),
            'previous_scores': float(data['previous_scores'])
        }

        df_input = pd.DataFrame([features])

        # Load scaler and transform features
        scaler = load_scaler()
        if scaler is None:
            return jsonify({'error': 'Scaler not found. Please train the model first.'})

        scaled_features = scaler.transform(df_input)

        # Load model and predict
        model = load_model()
        if model is None:
            return jsonify({'error': 'Model not found. Please train the model first.'})

        prediction = model.predict(scaled_features)[0]

        # Clamp prediction to valid range [0, 100]
        prediction = np.clip(prediction, 0, 100)

        return jsonify({
            'prediction': round(float(prediction), 2),
            'success': True
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/train', methods=['POST'])
def train_model():
    """Train the model from data and save it."""
    try:
        # Load data
        df = pd.read_csv('data/exams.csv')

        # Extract features and target
        X = df.drop(columns=["student_id", "exam_score"])
        y = df["exam_score"]

        # Preprocess
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train Linear Regression (simpler and often more accurate for this data)
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_scaled, y)

        # Save model and scaler
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)

        return jsonify({
            'success': True,
            'message': 'Model trained and saved successfully!'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
