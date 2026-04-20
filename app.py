from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import subprocess
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# ── FFMPEG PATH (Windows) ────────────────────────────────────────────────────
FFMPEG  = "ffmpeg"
FFPROBE = "ffprobe"
# ── Optional voice/audio imports ─────────────────────────────────────────────
try:
    import pickle
    import librosa
    import librosa.feature
    VOICE_LIBS_AVAILABLE = True
except ImportError:
    VOICE_LIBS_AVAILABLE = False
    print("[WARN] Voice libraries not installed. Voice prediction disabled.")

# PyTorch is optional now — we don't use the CNN anymore
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

MODEL_PATH      = "model.joblib"
NLP_MODEL_PATH  = "nlp_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PHYSIOLOGICAL MODEL
# ─────────────────────────────────────────────────────────────────────────────
def generate_synthetic_data(n=4000, random_state=42):
    np.random.seed(random_state)
    snoring     = np.random.uniform(0, 100, n)
    body_temp   = np.random.uniform(95, 103, n)
    hours_sleep = np.random.uniform(0.5, 10, n)
    hr          = np.random.uniform(45, 220, n)
    rule = (
        (snoring > 70).astype(int) +
        (body_temp > 99).astype(int) +
        (hours_sleep < 5).astype(int) +
        (hr > 100).astype(int)
    )
    label = (rule >= 2).astype(int)
    return pd.DataFrame({
        "snoring_range": snoring, "body_temperature": body_temp,
        "hours_sleep": hours_sleep, "heart_rate": hr, "stress": label
    })

def train_and_save_model(path=MODEL_PATH):
    print("Training physiological model on synthetic data...")
    df = generate_synthetic_data()
    X  = df[["snoring_range", "body_temperature", "hours_sleep", "heart_rate"]]
    y  = df["stress"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Physiological model trained. Accuracy: {acc:.3f}")
    joblib.dump(clf, path)
    return clf

def load_rf_model():
    if os.path.exists(MODEL_PATH):
        print("Loading existing physiological model...")
        return joblib.load(MODEL_PATH)
    return train_and_save_model()

rf_model = load_rf_model()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  NLP MODEL (text only, CNN removed)
# ─────────────────────────────────────────────────────────────────────────────
nlp_model  = None
vectorizer = None

if VOICE_LIBS_AVAILABLE:
    if os.path.exists(VECTORIZER_PATH) and os.path.exists(NLP_MODEL_PATH):
        try:
            vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))
            nlp_model  = pickle.load(open(NLP_MODEL_PATH, "rb"))
            print("[OK] NLP text model loaded.")
        except Exception as e:
            print(f"[WARN] Could not load NLP model: {e}")

VOICE_ENABLED = VOICE_LIBS_AVAILABLE
print(f"[INFO] Voice prediction: {'ENABLED (audio features)' if VOICE_ENABLED else 'DISABLED'}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  AUDIO-FEATURE-BASED STRESS DETECTION  (replaces CNN)
#     Uses real acoustic correlates of stress from speech science:
#     - Pitch (F0): stressed speech has higher & more variable pitch
#     - Energy (RMS): stressed speech is louder with more variation
#     - Speaking rate (ZCR proxy): faster/irregular in stress
#     - Spectral centroid: higher in stressed speech (more high freq energy)
#     - MFCCs variance: more variable under stress
# ─────────────────────────────────────────────────────────────────────────────
def extract_audio_features(audio_bytes, sr=16000):
    import tempfile, uuid

    tmp_path = os.path.join(tempfile.gettempdir(), f"stress_{uuid.uuid4().hex}.webm")
    wav_path = tmp_path.replace('.webm', '.wav')

    try:
        with open(tmp_path, 'wb') as f:
            f.write(audio_bytes)

        result = subprocess.run(
            [FFMPEG, "-y", "-i", tmp_path, "-ar", str(sr), wav_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {result.stderr.decode()}")

        y, _ = librosa.load(wav_path, sr=sr)

    finally:
        for p in [tmp_path, wav_path]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except:
                pass

    if len(y) < sr * 0.3:
        raise ValueError("Audio too short (< 0.3 seconds). Please speak longer.")

    features = {}

    # ── Pitch (F0) ────────────────────────────────────────────────────────────
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'), sr=sr
    )
    voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
    if len(voiced_f0) > 5:
        features['pitch_mean']   = float(np.mean(voiced_f0))
        features['pitch_std']    = float(np.std(voiced_f0))
        features['pitch_range']  = float(np.ptp(voiced_f0))   # max-min
        features['voiced_ratio'] = float(np.sum(voiced_flag) / len(voiced_flag))
    else:
        features['pitch_mean']   = 150.0
        features['pitch_std']    = 20.0
        features['pitch_range']  = 40.0
        features['voiced_ratio'] = 0.4

    # ── Energy (RMS) ──────────────────────────────────────────────────────────
    rms = librosa.feature.rms(y=y)[0]
    features['rms_mean'] = float(np.mean(rms))
    features['rms_std']  = float(np.std(rms))
    features['rms_cv']   = float(np.std(rms) / (np.mean(rms) + 1e-9))  # coeff variation

    # ── Spectral centroid (brightness) ────────────────────────────────────────
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['spectral_centroid_mean'] = float(np.mean(sc))
    features['spectral_centroid_std']  = float(np.std(sc))

    # ── Zero-crossing rate (speaking rate proxy) ──────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zcr_mean'] = float(np.mean(zcr))
    features['zcr_std']  = float(np.std(zcr))

    # ── MFCC variability ──────────────────────────────────────────────────────
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features['mfcc_var_mean'] = float(np.mean(np.var(mfcc, axis=1)))

    # ── Tempo ─────────────────────────────────────────────────────────────────
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = float(tempo) if not np.isnan(tempo) else 120.0

    return features, y, sr


def score_stress_from_features(features):
    """
    Rule-based scoring grounded in speech stress research.
    Returns (stress_prob, not_stressed_prob, detail_dict)
    """
    score = 0.0
    max_score = 0.0
    detail = {}

    # ── 1. Pitch mean (higher = more stressed, typical range 80-300 Hz) ───────
    pm = features['pitch_mean']
    max_score += 20
    if pm > 220:
        pts = 20
    elif pm > 180:
        pts = 14
    elif pm > 140:
        pts = 8
    else:
        pts = 3
    score += pts
    detail['pitch_mean_score'] = pts

    # ── 2. Pitch variability (high std = stressed/emotional speech) ───────────
    ps = features['pitch_std']
    max_score += 20
    if ps > 60:
        pts = 20
    elif ps > 40:
        pts = 14
    elif ps > 25:
        pts = 8
    else:
        pts = 3
    score += pts
    detail['pitch_std_score'] = pts

    # ── 3. RMS coefficient of variation (erratic loudness = stress) ──────────
    rcv = features['rms_cv']
    max_score += 15
    if rcv > 1.0:
        pts = 15
    elif rcv > 0.7:
        pts = 10
    elif rcv > 0.4:
        pts = 6
    else:
        pts = 2
    score += pts
    detail['rms_cv_score'] = pts

    # ── 4. Spectral centroid (brighter/higher = more stressed) ───────────────
    scm = features['spectral_centroid_mean']
    max_score += 15
    if scm > 3000:
        pts = 15
    elif scm > 2000:
        pts = 10
    elif scm > 1200:
        pts = 6
    else:
        pts = 2
    score += pts
    detail['spectral_centroid_score'] = pts

    # ── 5. ZCR mean (higher crossing = faster/more tense articulation) ────────
    zm = features['zcr_mean']
    max_score += 15
    if zm > 0.12:
        pts = 15
    elif zm > 0.08:
        pts = 10
    elif zm > 0.05:
        pts = 6
    else:
        pts = 2
    score += pts
    detail['zcr_score'] = pts

    # ── 6. MFCC variance (more spectral change = more emotional speech) ───────
    mv = features['mfcc_var_mean']
    max_score += 15
    if mv > 200:
        pts = 15
    elif mv > 100:
        pts = 10
    elif mv > 50:
        pts = 6
    else:
        pts = 2
    score += pts
    detail['mfcc_var_score'] = pts

    # ── Normalise to probability ──────────────────────────────────────────────
    raw_prob = score / max_score          # 0..1
    # Apply a mild sigmoid stretch so mid-values don't all cluster around 0.5
    import math
    stretched = 1 / (1 + math.exp(-8 * (raw_prob - 0.5)))
    stress_prob      = round(stretched, 4)
    not_stress_prob  = round(1.0 - stretched, 4)

    return stress_prob, not_stress_prob, detail


def predict_audio_stress(audio_bytes):
    """Main entry: extract features → score → return probs + debug info."""
    features, y, sr = extract_audio_features(audio_bytes)
    stress_p, not_stress_p, detail = score_stress_from_features(features)
    return stress_p, not_stress_p, features, detail


def predict_nlp_text(text):
    if not text or not text.strip() or vectorizer is None or nlp_model is None:
        return None, None
    vec  = vectorizer.transform([text.strip()])
    prob = nlp_model.predict_proba(vec)[0]
    return float(prob[1]), float(prob[0])


def combine_voice_scores(audio_s, audio_ns, nlp_s, nlp_ns):
    if nlp_s is None:
        return audio_s, audio_ns, "audio_features_only"
    return (0.6*audio_s + 0.4*nlp_s), (0.6*audio_ns + 0.4*nlp_ns), "audio+nlp"


# ─────────────────────────────────────────────────────────────────────────────
# 4.  ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_physiological():
    data = request.get_json(force=True)
    try:
        snoring     = float(data.get("snoring_range", 0))
        body_temp   = float(data.get("body_temperature", 0))
        sleep_hours = float(data.get("hours_sleep", 0))
        heart_rate  = float(data.get("heart_rate", 0))
    except Exception:
        return jsonify({"error": "Invalid inputs. Ensure numeric values."}), 400

    X    = np.array([[snoring, body_temp, sleep_hours, heart_rate]])
    pred = rf_model.predict(X)[0]
    prob = float(rf_model.predict_proba(X)[0][pred])
    label = "High" if pred == 1 else "Low"

    return jsonify({
        "stress_level": label,
        "probability" : round(prob, 3),
        "message"     : f"Your Stress level is {label}"
    })


@app.route("/predict-voice", methods=["POST"])
def predict_voice():
    import traceback
    if not VOICE_ENABLED:
        return jsonify({"error": "Voice libraries (librosa) not available on this server."}), 503
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file sent"}), 400

        audio_bytes = request.files["audio"].read()
        transcript  = request.form.get("transcript", "").strip()
        print(f"[DEBUG] Audio: {len(audio_bytes)} bytes | Transcript: '{transcript}'")

        # ── Audio feature-based prediction (replaces CNN) ─────────────────
        audio_s, audio_ns, features, feat_detail = predict_audio_stress(audio_bytes)

        # ── NLP text prediction (optional) ────────────────────────────────
        nlp_s, nlp_ns = predict_nlp_text(transcript)

        # ── Combine ───────────────────────────────────────────────────────
        final_s, final_ns, method = combine_voice_scores(audio_s, audio_ns, nlp_s, nlp_ns)

        label = "Stressed" if final_s > final_ns else "Not Stressed"

        return jsonify({
            "prediction"   : label,
            "stress_level" : label,
            "confidence"   : {"stressed": round(final_s, 4), "not_stressed": round(final_ns, 4)},
            "audio_score"  : {"stressed": round(audio_s, 4), "not_stressed": round(audio_ns, 4)},
            "nlp_score"    : {
                "stressed":     round(nlp_s, 4) if nlp_s is not None else None,
                "not_stressed": round(nlp_ns, 4) if nlp_ns is not None else None
            },
            "acoustic_features": {
                "pitch_mean_hz"      : round(features.get("pitch_mean", 0), 1),
                "pitch_std_hz"       : round(features.get("pitch_std", 0), 1),
                "pitch_range_hz"     : round(features.get("pitch_range", 0), 1),
                "voiced_ratio"       : round(features.get("voiced_ratio", 0), 3),
                "rms_mean"           : round(features.get("rms_mean", 0), 5),
                "rms_variability"    : round(features.get("rms_cv", 0), 3),
                "spectral_centroid"  : round(features.get("spectral_centroid_mean", 0), 1),
                "zcr_mean"           : round(features.get("zcr_mean", 0), 5),
                "mfcc_variance"      : round(features.get("mfcc_var_mean", 0), 2),
                "tempo_bpm"          : round(features.get("tempo", 0), 1),
            },
            "transcript"   : transcript or None,
            "method"       : method,
            "message"      : f"Voice Analysis: {label}"
        })
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return jsonify({"error": str(e), "traceback": tb}), 500


@app.route("/voice-status")
def voice_status():
    return jsonify({"voice_enabled": VOICE_ENABLED})


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8000)
