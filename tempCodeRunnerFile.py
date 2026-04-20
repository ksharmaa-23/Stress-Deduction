


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
# 2.  VOICE / CNN MODEL
# ─────────────────────────────────────────────────────────────────────────────
cnn_model  = None
nlp_model  = None
vectorizer = None

if VOICE_LIBS_AVAILABLE:
    class StressCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1   = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.conv2   = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.pool    = nn.AdaptiveAvgPool2d((10, 10))
            self.fc1     = nn.Linear(32 * 10 * 10, 64)
            self.fc2     = nn.Linear(64, 2)
            self.relu    = nn.ReLU()
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(self.relu(self.fc1(x)))
            return self.fc2(x)

    if os.path.exists(CNN_MODEL_PATH):
        try:
            cnn_model = StressCNN()
            cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location="cpu"))
            cnn_model.eval()
            print("[OK] CNN voice model loaded.")
        except Exception as e:
            print(f"[WARN] Could not load CNN model: {e}")
            cnn_model = None

    if os.path.exists(VECTORIZER_PATH) and os.path.exists(NLP_MODEL_PATH):
        try:
            vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))
            nlp_model  = pickle.load(open(NLP_MODEL_PATH, "rb"))
            print("[OK] NLP text model loaded.")
        except Exception as e:
            print(f"[WARN] Could not load NLP model: {e}")

VOICE_ENABLED = cnn_model is not None
print(f"[INFO] Voice prediction: {'ENABLED' if VOICE_ENABLED else 'DISABLED (model files not found)'}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  HELPERS FOR VOICE
# ─────────────────────────────────────────────────────────────────────────────
def extract_mfcc(audio_bytes, sr=16000, n_mfcc=40, max_len=100):
    import tempfile, uuid

    tmp_path = os.path.join(tempfile.gettempdir(), f"stress_{uuid.uuid4().hex}.webm")
    wav_path = tmp_path.replace('.webm', '.wav')

    try:
        with open(tmp_path, 'wb') as f:
            f.write(audio_bytes)

        # Call ffmpeg directly using full path — bypasses pydub detection
        result = subprocess.run(
            [FFMPEG, "-y", "-i", tmp_path, wav_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {result.stderr.decode()}")

        y, _ = librosa.load(wav_path, sr=sr)

    finally:
        try:
            if os.path.exists(tmp_path): os.remove(tmp_path)
        except: pass
        try:
            if os.path.exists(wav_path): os.remove(wav_path)
        except: pass

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

def predict_cnn(audio_bytes):
    mfcc   = extract_mfcc(audio_bytes)
    tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(cnn_model(tensor), dim=1).squeeze()
    return float(probs[1]), float(probs[0])

def predict_nlp_text(text):
    if not text or not text.strip() or vectorizer is None or nlp_model is None:
        return None, None
    vec  = vectorizer.transform([text.strip()])
    prob = nlp_model.predict_proba(vec)[0]
    return float(prob[1]), float(prob[0])

def combine_voice_scores(cnn_s, cnn_ns, nlp_s, nlp_ns):
    if nlp_s is None:
        return cnn_s, cnn_ns, "cnn_only"
    return (0.5*cnn_s + 0.5*nlp_s), (0.5*cnn_ns + 0.5*nlp_ns), "cnn+nlp"


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
        return jsonify({"error": "Voice model not available on this server."}), 503
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file sent"}), 400

        audio_bytes = request.files["audio"].read()
        transcript  = request.form.get("transcript", "").strip()
        print(f"[DEBUG] Audio: {len(audio_bytes)} bytes | Transcript: '{transcript}'")

        cnn_s, cnn_ns             = predict_cnn(audio_bytes)
        nlp_s, nlp_ns             = predict_nlp_text(transcript)
        final_s, final_ns, method = combine_voice_scores(cnn_s, cnn_ns, nlp_s, nlp_ns)

        label = "Stressed" if final_s > final_ns else "Not Stressed"
        return jsonify({
            "prediction"  : label,
            "stress_level": label,
            "confidence"  : {"stressed": round(final_s, 4), "not_stressed": round(final_ns, 4)},
            "cnn_score"   : {"stressed": round(cnn_s, 4),   "not_stressed": round(cnn_ns, 4)},
            "nlp_score"   : {
                "stressed":     round(nlp_s, 4) if nlp_s is not None else None,
                "not_stressed": round(nlp_ns, 4) if nlp_ns is not None else None
            },
            "transcript"  : transcript or None,
            "method"      : method,
            "message"     : f"Voice Analysis: {label}"
        })
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return jsonify({"error": str(e), "traceback": tb}), 500


@app.route("/voice-status")
def voice_status():
    return jsonify({"voice_enabled": VOICE_ENABLED})


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8000)