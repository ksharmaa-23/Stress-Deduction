from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

MODEL_PATH = "model.joblib"

app = Flask(__name__, static_folder="static", template_folder="templates")

def generate_synthetic_data(n=4000, random_state=42):
    np.random.seed(random_state)
    # features:
    # snoring_range: 0-100
    # body_temperature: 95 - 103 Fahrenheit
    # hours_sleep: 0.5 - 10
    # heart_rate: 45 - 220
    snoring = np.random.uniform(0, 100, n)
    body_temp = np.random.uniform(95, 103, n)
    hours_sleep = np.random.uniform(0.5, 10, n)
    hr = np.random.uniform(45, 220, n)

    # heuristic label: 1 = High stress, 0 = Low stress
    # produce some realistic noise
    rule = (
        (snoring > 70).astype(int) * 1 +
        (body_temp > 99).astype(int) * 1 +
        (hours_sleep < 5).astype(int) * 1 +
        (hr > 100).astype(int) * 1
    )
    # if rule >= 2 -> High stress
    label = (rule >= 2).astype(int)

    df = pd.DataFrame({
        "snoring_range": snoring,
        "body_temperature": body_temp,
        "hours_sleep": hours_sleep,
        "heart_rate": hr,
        "stress": label
    })
    return df

def train_and_save_model(path=MODEL_PATH):
    print("Training model on synthetic data...")
    df = generate_synthetic_data()
    X = df[["snoring_range", "body_temperature", "hours_sleep", "heart_rate"]]
    y = df["stress"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model trained. Test accuracy: {acc:.3f}")
    joblib.dump(clf, path)
    return clf

def load_model(path=MODEL_PATH):
    if os.path.exists(path):
        print("Loading existing model...")
        return joblib.load(path)
    else:
        return train_and_save_model(path)

model = load_model()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    # expected keys: snoring_range, body_temperature, hours_sleep, heart_rate
    try:
        snoring = float(data.get("snoring_range", 0))
        body_temp = float(data.get("body_temperature", 0))
        sleep_hours = float(data.get("hours_sleep", 0))
        heart_rate = float(data.get("heart_rate", 0))
    except Exception as e:
        return jsonify({"error": "Invalid inputs. Ensure numeric values."}), 400

    X = np.array([[snoring, body_temp, sleep_hours, heart_rate]])
    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    prob = float(probs[pred])

    label = "High" if pred == 1 else "Low"
    message = f"Your Stress level is {label}"
    return jsonify({
        "stress_level": label,
        "probability": round(prob, 3),
        "message": message
    })

if __name__ == "__main__":
    # runs on localhost:8000 by default; you can change port if needed
    app.run(debug=True, host="127.0.0.1", port=8000)
