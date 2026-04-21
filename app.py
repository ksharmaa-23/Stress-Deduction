from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import io
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

try:
    import soundfile as sf
    VOICE_LIBS_AVAILABLE = True
except ImportError:
    VOICE_LIBS_AVAILABLE = False

MODEL_PATH       = "model.joblib"
VOICE_MODEL_PATH = "voice_model.joblib"
SCALER_PATH      = "voice_scaler.joblib"

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)


def train_physiological_model():
    csv_path = "SaYoPillow.csv"
    if os.path.exists(csv_path):
        print("[INFO] Training physiological model from CSV...")
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        feature_cols = [c for c in df.columns if c != 'stress_level']
        X = df[feature_cols]
        y = (df['stress_level'] >= 2).astype(int)
    else:
        print("[INFO] CSV not found, using synthetic data...")
        np.random.seed(42)
        n = 4000
        X = pd.DataFrame({
            'sr':  np.random.uniform(0, 100, n),
            'rr':  np.random.uniform(10, 30, n),
            't':   np.random.uniform(95, 103, n),
            'lm':  np.random.uniform(0, 20, n),
            'bo':  np.random.uniform(85, 100, n),
            'rem': np.random.uniform(0, 25, n),
            'hr':  np.random.uniform(45, 120, n),
            'sl':  np.random.uniform(0.5, 10, n),
        })
        rule = ((X['sr'] > 70).astype(int) + (X['t'] > 99).astype(int) +
                (X['sl'] < 5).astype(int) + (X['hr'] > 100).astype(int))
        y = (rule >= 2).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"[INFO] Physiological model accuracy: {acc:.3f}")
    joblib.dump({'model': clf, 'features': list(X.columns)}, MODEL_PATH)
    return clf, list(X.columns)


def load_physiological_model():
    if os.path.exists(MODEL_PATH):
        data = joblib.load(MODEL_PATH)
        return data['model'], data['features']
    return train_physiological_model()


phys_model, phys_features = load_physiological_model()


def train_voice_model():
    csv_path = "voice_dataset.csv"
    if os.path.exists(csv_path):
        print("[INFO] Training voice model from CSV...")
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        feature_cols = [c for c in df.columns if c != 'stress_label']
        X = df[feature_cols].values
        y = df['stress_label'].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        print(f"[INFO] Voice model accuracy: {acc:.3f}")
        joblib.dump(clf, VOICE_MODEL_PATH)
        joblib.dump({'scaler': scaler, 'features': list(feature_cols)}, SCALER_PATH)
        return clf, scaler, list(feature_cols)
    return None, None, None


def load_voice_model():
    if os.path.exists(VOICE_MODEL_PATH) and os.path.exists(SCALER_PATH):
        clf = joblib.load(VOICE_MODEL_PATH)
        data = joblib.load(SCALER_PATH)
        return clf, data['scaler'], data['features']
    return train_voice_model()


voice_model, voice_scaler, voice_features = load_voice_model()
VOICE_ENABLED = VOICE_LIBS_AVAILABLE and voice_model is not None
print(f"[INFO] Voice model: {'ENABLED' if VOICE_ENABLED else 'DISABLED'}")


def extract_audio_features(audio_bytes):
    try:
        audio_io = io.BytesIO(audio_bytes)
        y, sr = sf.read(audio_io, dtype='float32')
        if len(y.shape) > 1:
            y = y.mean(axis=1)
    except Exception as e:
        print(f"[WARN] Audio read error: {e}")
        return None

    if len(y) < 1000:
        raise ValueError("Audio too short. Please speak for at least 1 second.")

    frame_size = 512
    frames = [y[i:i+frame_size] for i in range(0, len(y)-frame_size, frame_size//2)]
    if not frames:
        return None

    rms_vals = np.array([np.sqrt(np.mean(f**2)) for f in frames])
    rms_cv = float(np.std(rms_vals) / (np.mean(rms_vals) + 1e-9))

    zcr_mean = float(np.mean([
        np.sum(np.abs(np.diff(np.sign(f)))) / (2 * frame_size)
        for f in frames
    ]))

    sc_vals = []
    for f in frames:
        spec  = np.abs(np.fft.rfft(f))
        freqs = np.fft.rfftfreq(frame_size, 1.0 / sr)
        sc_vals.append(np.sum(freqs * spec) / (np.sum(spec) + 1e-9))
    spectral_centroid_mean = float(np.mean(sc_vals))

    chunk   = y[:min(len(y), sr)]
    corr    = np.correlate(chunk, chunk, mode='full')[len(chunk)-1:]
    min_lag = int(sr / 400)
    max_lag = int(sr / 70)
    if max_lag < len(corr) and min_lag < max_lag:
        peak_lag   = np.argmax(corr[min_lag:max_lag]) + min_lag
        pitch_mean = float(sr / peak_lag) if peak_lag > 0 else 150.0
    else:
        pitch_mean = 150.0

    pitch_std   = rms_cv * 30
    mfcc_var    = float(np.var(y) * 1000)
    speech_rate = zcr_mean

    return {
        'pitch_mean':             pitch_mean,
        'pitch_std':              pitch_std,
        'rms_cv':                 rms_cv,
        'zcr_mean':               zcr_mean,
        'spectral_centroid_mean': spectral_centroid_mean,
        'mfcc_var_mean':          mfcc_var,
        'speech_rate':            speech_rate,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_physiological():
    data = request.get_json(force=True)
    try:
        key_map = {
            'sr':  ['snoring_range', 'sr'],
            'rr':  ['respiration_rate', 'rr'],
            't':   ['body_temperature', 't'],
            'lm':  ['limb_movement', 'lm'],
            'bo':  ['blood_oxygen', 'bo'],
            'rem': ['eye_movement', 'rem'],
            'hr':  ['heart_rate', 'hr'],
            'sl':  ['hours_sleep', 'sl'],
        }
        row = {}
        for feat in phys_features:
            val = None
            if feat in key_map:
                for k in key_map[feat]:
                    if k in data:
                        val = float(data[k])
                        break
            if val is None and feat in data:
                val = float(data[feat])
            row[feat] = val if val is not None else 0.0

        X     = pd.DataFrame([row])[phys_features]
        pred  = phys_model.predict(X)[0]
        prob  = float(phys_model.predict_proba(X)[0][pred])
        label = "High" if pred == 1 else "Low"
        return jsonify({
            "stress_level": label,
            "probability":  round(prob, 3),
            "message":      f"Your Stress level is {label}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict-voice", methods=["POST"])
def predict_voice():
    import traceback
    if not VOICE_ENABLED:
        return jsonify({"error": "Voice model not available."}), 503
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file sent"}), 400

        audio_bytes = request.files["audio"].read()
        features    = extract_audio_features(audio_bytes)

        if features is None:
            return jsonify({"error": "Could not process audio. Please try again."}), 400

        feat_order = voice_features if voice_features else [
            'pitch_mean', 'pitch_std', 'rms_cv', 'zcr_mean',
            'spectral_centroid_mean', 'mfcc_var_mean', 'speech_rate'
        ]
        feat_vec    = np.array([[features.get(f, 0.0) for f in feat_order]])
        feat_scaled = voice_scaler.transform(feat_vec)

        pred  = voice_model.predict(feat_scaled)[0]
        proba = voice_model.predict_proba(feat_scaled)[0]
        label = "Stressed" if pred == 1 else "Not Stressed"

        stressed_prob     = float(proba[1]) if len(proba) > 1 else float(pred)
        not_stressed_prob = float(proba[0]) if len(proba) > 1 else 1 - float(pred)

        return jsonify({
            "prediction":   label,
            "stress_level": label,
            "confidence": {
                "stressed":     round(stressed_prob, 4),
                "not_stressed": round(not_stressed_prob, 4)
            },
            "audio_score": {
                "stressed":     round(stressed_prob, 4),
                "not_stressed": round(not_stressed_prob, 4)
            },
            "nlp_score": {"stressed": None, "not_stressed": None},
            "acoustic_features": {
                "pitch_mean_hz":     round(features.get("pitch_mean", 0), 1),
                "pitch_std_hz":      round(features.get("pitch_std", 0), 1),
                "rms_variability":   round(features.get("rms_cv", 0), 3),
                "spectral_centroid": round(features.get("spectral_centroid_mean", 0), 1),
                "zcr_mean":          round(features.get("zcr_mean", 0), 5),
                "mfcc_variance":     round(features.get("mfcc_var_mean", 0), 2),
            },
            "method":  "csv_voice_model",
            "message": f"Voice Analysis: {label}"
        })
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/voice-status")
def voice_status():
    return jsonify({"voice_enabled": VOICE_ENABLED})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=False, host="0.0.0.0", port=port)