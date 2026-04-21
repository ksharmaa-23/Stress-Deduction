from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import io
import wave
import struct
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

MODEL_PATH       = "model.joblib"
VOICE_MODEL_PATH = "voice_model.joblib"
SCALER_PATH      = "voice_scaler.joblib"

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)


# ─────────────────────────────────────────────────────────────────────────────
# 1. PHYSIOLOGICAL MODEL
# ─────────────────────────────────────────────────────────────────────────────
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
        print("Training physiological model on synthetic data...")
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


# ─────────────────────────────────────────────────────────────────────────────
# 2. VOICE MODEL
# ─────────────────────────────────────────────────────────────────────────────
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
        clf  = joblib.load(VOICE_MODEL_PATH)
        data = joblib.load(SCALER_PATH)
        return clf, data['scaler'], data['features']
    return train_voice_model()

voice_model, voice_scaler, voice_features = load_voice_model()
VOICE_ENABLED = voice_model is not None
print(f"[INFO] Voice model: {'ENABLED' if VOICE_ENABLED else 'DISABLED'}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. AUDIO READING  (robust multi-method)
# ─────────────────────────────────────────────────────────────────────────────
def read_audio_bytes(audio_bytes):
    """Try multiple methods to read audio bytes → (samples_float32, sample_rate)."""

    # Method 1: Python built-in wave (standard PCM WAV)
    try:
        buf = io.BytesIO(audio_bytes)
        with wave.open(buf, 'rb') as wf:
            n_channels = wf.getnchannels()
            sampwidth  = wf.getsampwidth()
            framerate  = wf.getframerate()
            n_frames   = wf.getnframes()
            raw        = wf.readframes(n_frames)

        if sampwidth == 2:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)

        if len(samples) > 0:
            print(f"[DEBUG] wave: {len(samples)} samples @ {framerate} Hz")
            return samples, framerate
    except Exception as e:
        print(f"[DEBUG] wave failed: {e}")

    # Method 2: Manual WAV chunk parser (handles quirky AudioContext output)
    try:
        data = audio_bytes
        fmt_idx = data.find(b'fmt ')
        data_idx = data.find(b'data')
        if fmt_idx >= 0 and data_idx >= 0:
            fmt_data = data[fmt_idx + 8: fmt_idx + 24]
            audio_format, n_ch, sample_rate, _, _, bits = struct.unpack('<HHIIHH', fmt_data)
            data_size  = struct.unpack('<I', data[data_idx + 4: data_idx + 8])[0]
            audio_data = data[data_idx + 8: data_idx + 8 + data_size]

            if bits == 16:
                samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            elif bits == 32:
                # Could be float32 or int32
                try:
                    samples = np.frombuffer(audio_data, dtype=np.float32)
                    if np.max(np.abs(samples)) > 1.5:   # looks like int32
                        samples = np.frombuffer(audio_data, dtype=np.int32).astype(np.float32) / 2147483648.0
                except Exception:
                    samples = np.frombuffer(audio_data, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                samples = np.frombuffer(audio_data, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

            if n_ch > 1:
                samples = samples.reshape(-1, n_ch).mean(axis=1)

            if len(samples) > 0:
                print(f"[DEBUG] manual WAV: {len(samples)} samples @ {sample_rate} Hz")
                return samples, sample_rate
    except Exception as e:
        print(f"[DEBUG] manual WAV parse failed: {e}")

    # Method 3: soundfile (handles webm/ogg/flac if libsndfile supports it)
    try:
        import soundfile as sf
        buf = io.BytesIO(audio_bytes)
        samples, sr = sf.read(buf, dtype='float32')
        if len(samples.shape) > 1:
            samples = samples.mean(axis=1)
        if len(samples) > 0:
            print(f"[DEBUG] soundfile: {len(samples)} samples @ {sr} Hz")
            return samples, sr
    except Exception as e:
        print(f"[DEBUG] soundfile failed: {e}")

    # Method 4: pydub (handles webm/opus if ffmpeg is present)
    try:
        from pydub import AudioSegment
        buf = io.BytesIO(audio_bytes)
        seg = AudioSegment.from_file(buf)
        seg = seg.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        raw = np.array(seg.get_array_of_samples(), dtype=np.int16)
        samples = raw.astype(np.float32) / 32768.0
        print(f"[DEBUG] pydub: {len(samples)} samples @ 16000 Hz")
        return samples, 16000
    except Exception as e:
        print(f"[DEBUG] pydub failed: {e}")

    print("[WARN] All audio reading methods failed")
    return None, None


def extract_audio_features(audio_bytes):
    y, sr = read_audio_bytes(audio_bytes)

    if y is None or len(y) < 100:
        return None

    sr = sr or 16000
    frame_size = min(512, max(64, len(y) // 8))
    hop    = frame_size // 2
    frames = [y[i: i + frame_size] for i in range(0, len(y) - frame_size, hop)]

    if not frames:
        return None

    # RMS energy
    rms_vals = np.array([np.sqrt(np.mean(f ** 2) + 1e-10) for f in frames])
    rms_cv   = float(np.std(rms_vals) / (np.mean(rms_vals) + 1e-9))

    # ZCR
    zcr_vals = [np.sum(np.abs(np.diff(np.sign(f)))) / (2 * len(f)) for f in frames]
    zcr_mean = float(np.mean(zcr_vals))

    # Spectral centroid
    sc_vals = []
    for f in frames:
        spec  = np.abs(np.fft.rfft(f))
        freqs = np.fft.rfftfreq(len(f), 1.0 / sr)
        sc_vals.append(float(np.sum(freqs * spec) / (np.sum(spec) + 1e-9)))
    spectral_centroid_mean = float(np.mean(sc_vals))

    # Pitch via autocorrelation
    chunk   = y[: min(len(y), sr)]
    corr    = np.correlate(chunk, chunk, mode='full')[len(chunk) - 1:]
    min_lag = max(1, int(sr / 400))
    max_lag = min(len(corr) - 1, int(sr / 70))
    if max_lag > min_lag:
        peak       = np.argmax(corr[min_lag: max_lag]) + min_lag
        pitch_mean = float(sr / peak) if peak > 0 else 150.0
    else:
        pitch_mean = 150.0

    return {
        'pitch_mean':             pitch_mean,
        'pitch_std':              rms_cv * 30,
        'rms_cv':                 rms_cv,
        'zcr_mean':               zcr_mean,
        'spectral_centroid_mean': spectral_centroid_mean,
        'mfcc_var_mean':          float(np.var(y) * 1000),
        'speech_rate':            zcr_mean,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_physiological():
    data = request.get_json(force=True)
    try:
        # Map both long-form and short-form field names to model features
        key_map = {
            'sr':  ['snoring_range',    'sr'],
            'rr':  ['respiration_rate', 'rr'],
            't':   ['body_temperature', 't'],
            'lm':  ['limb_movement',    'lm'],
            'bo':  ['blood_oxygen',     'bo'],
            'rem': ['eye_movement',     'rem'],
            'hr':  ['heart_rate',       'hr'],
            'sl':  ['hours_sleep',      'sl'],
        }
        # Sensible defaults so prediction is not distorted by missing fields
        defaults = {
            'sr': 30, 'rr': 15, 't': 98.6, 'lm': 8,
            'bo': 97, 'rem': 12, 'hr': 72, 'sl': 7
        }
        row = {}
        for feat in phys_features:
            val = None
            for k in key_map.get(feat, [feat]):
                if k in data:
                    val = float(data[k])
                    break
            row[feat] = val if val is not None else defaults.get(feat, 0.0)

        X    = pd.DataFrame([row])[phys_features]
        pred = phys_model.predict(X)[0]
        prob = float(phys_model.predict_proba(X)[0][pred])
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
        print(f"[DEBUG] Audio received: {len(audio_bytes)} bytes, header: {audio_bytes[:4]}")

        features = extract_audio_features(audio_bytes)
        if features is None:
            return jsonify({
                "error": "Could not process audio. Please speak for at least 2 seconds and try again."
            }), 400

        feat_order = voice_features or [
            'pitch_mean', 'pitch_std', 'rms_cv', 'zcr_mean',
            'spectral_centroid_mean', 'mfcc_var_mean', 'speech_rate'
        ]
        feat_vec    = np.array([[features.get(f, 0.0) for f in feat_order]])
        feat_scaled = voice_scaler.transform(feat_vec)

        pred  = voice_model.predict(feat_scaled)[0]
        proba = voice_model.predict_proba(feat_scaled)[0]
        label = "Stressed" if pred == 1 else "Not Stressed"
        sp    = float(proba[1]) if len(proba) > 1 else float(pred)
        nsp   = float(proba[0]) if len(proba) > 1 else 1 - float(pred)

        return jsonify({
            "prediction":   label,
            "stress_level": label,
            "confidence":   {"stressed": round(sp, 4), "not_stressed": round(nsp, 4)},
            "audio_score":  {"stressed": round(sp, 4), "not_stressed": round(nsp, 4)},
            "nlp_score":    {"stressed": None, "not_stressed": None},
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
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/voice-status")
def voice_status():
    return jsonify({"voice_enabled": VOICE_ENABLED})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=False, host="0.0.0.0", port=port)
