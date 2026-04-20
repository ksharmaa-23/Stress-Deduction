// ── TAB SWITCHING ────────────────────────────────────────────────────────────
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
  });
});


// ── PHYSIOLOGICAL PREDICTION ──────────────────────────────────────────────────
document.getElementById("predictBtn").addEventListener("click", async () => {
  const snoring = document.getElementById("snoring").value;
  const temp    = document.getElementById("temp").value;
  const hours   = document.getElementById("hours").value;
  const hr      = document.getElementById("hr").value;

  const payload = {
    snoring_range    : Number(snoring),
    body_temperature : Number(temp),
    hours_sleep      : Number(hours),
    heart_rate       : Number(hr)
  };

  const resultEl = document.getElementById("result");
  resultEl.textContent = "Analyzing...";
  resultEl.className   = "result-box show";

  try {
    const res  = await fetch("/predict", {
      method : "POST",
      headers: { "Content-Type": "application/json" },
      body   : JSON.stringify(payload)
    });

    if (!res.ok) {
      const err = await res.json();
      resultEl.textContent = err.error || "Prediction failed.";
      resultEl.className   = "result-box show error";
      return;
    }

    const data = await res.json();
    resultEl.textContent = `${data.message}  (${Math.round(data.probability * 100)}% confidence)`;
    resultEl.className   = "result-box show " + (data.stress_level === "High" ? "high" : "low");
  } catch (err) {
    resultEl.textContent = "Network error: " + err.message;
    resultEl.className   = "result-box show error";
  }
});


// ── VOICE RECORDING + ANALYSIS ────────────────────────────────────────────────
let mediaRecorder  = null;
let audioChunks    = [];
let recordedBlob   = null;
let timerInterval  = null;
let elapsedSeconds = 0;
let recognition    = null;
let transcript     = "";

const recordBtn      = document.getElementById("recordBtn");
const recordLabel    = document.getElementById("recordLabel");
const recordTimer    = document.getElementById("recordTimer");
const timerCount     = document.getElementById("timerCount");
const audioPlayback  = document.getElementById("audioPlayback");
const audioPlayer    = document.getElementById("audioPlayer");
const transcriptBox  = document.getElementById("transcriptBox");
const transcriptText = document.getElementById("transcriptText");
const analyzeBtn     = document.getElementById("analyzeBtn");
const voiceResult    = document.getElementById("voiceResult");
const voiceDetails   = document.getElementById("voiceDetails");

// Speech Recognition
function startSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) return;
  recognition = new SpeechRecognition();
  recognition.continuous     = true;
  recognition.interimResults = true;
  recognition.lang           = "en-US";
  recognition.onresult = (e) => {
    let interim = "";
    for (let i = e.resultIndex; i < e.results.length; i++) {
      if (e.results[i].isFinal) transcript += e.results[i][0].transcript + " ";
      else interim += e.results[i][0].transcript;
    }
    transcriptText.value = (transcript + interim).trim();
  };
  recognition.onerror = () => {};
  recognition.start();
}

function stopSpeechRecognition() {
  if (recognition) { try { recognition.stop(); } catch(_) {} recognition = null; }
}

function startTimer() {
  elapsedSeconds = 0;
  timerCount.textContent = "0";
  recordTimer.classList.remove("hidden");
  timerInterval = setInterval(() => {
    elapsedSeconds++;
    timerCount.textContent = elapsedSeconds;
  }, 1000);
}

function stopTimer() {
  clearInterval(timerInterval);
  timerInterval = null;
}

recordBtn.addEventListener("click", async () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    stopSpeechRecognition();
    stopTimer();
    recordBtn.classList.remove("recording");
    recordLabel.textContent = "Start Recording";
    recordBtn.querySelector(".rec-icon").textContent = "🎙️";
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    transcript  = "";
    transcriptText.value = "";

    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
    mediaRecorder.onstop = () => {
      recordedBlob = new Blob(audioChunks, { type: "audio/webm" });
      audioPlayer.src = URL.createObjectURL(recordedBlob);
      audioPlayback.classList.remove("hidden");
      transcriptBox.classList.remove("hidden");
      analyzeBtn.classList.remove("hidden");
      recordTimer.classList.add("hidden");
      stream.getTracks().forEach(t => t.stop());
    };

    mediaRecorder.start(250);
    startTimer();
    startSpeechRecognition();
    recordBtn.classList.add("recording");
    recordLabel.textContent = "Stop Recording";
    recordBtn.querySelector(".rec-icon").textContent = "⏹️";

    setTimeout(() => {
      if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        stopSpeechRecognition();
        stopTimer();
        recordBtn.classList.remove("recording");
        recordLabel.textContent = "Start Recording";
        recordBtn.querySelector(".rec-icon").textContent = "🎙️";
      }
    }, 60000);

  } catch (err) {
    voiceResult.textContent = "Microphone access denied: " + err.message;
    voiceResult.className   = "result-box show error";
  }
});

analyzeBtn.addEventListener("click", async () => {
  if (!recordedBlob) {
    voiceResult.textContent = "Please record audio first.";
    voiceResult.className   = "result-box show error";
    return;
  }

  voiceResult.textContent = "Analyzing your voice...";
  voiceResult.className   = "result-box show";
  voiceDetails.classList.add("hidden");
  analyzeBtn.disabled = true;

  try {
    const formData = new FormData();
    formData.append("audio",      recordedBlob, "recording.webm");
    formData.append("transcript", transcriptText.value.trim());

    const res = await fetch("/predict-voice", { method: "POST", body: formData });

    if (!res.ok) {
      const err = await res.json();
      voiceResult.textContent = res.status === 503
        ? "⚠️ Voice model not available. Use the Physiological tab."
        : (err.error || "Voice analysis failed.");
      voiceResult.className = "result-box show error";
      return;
    }

    const data = await res.json();
    const isStressed = data.prediction === "Stressed";
    const conf = isStressed
      ? (data.confidence.stressed * 100).toFixed(1)
      : (data.confidence.not_stressed * 100).toFixed(1);

    voiceResult.textContent = `${data.message}  (${conf}% confidence)`;
    voiceResult.className   = "result-box show " + (isStressed ? "high" : "low");

    // ── UPDATED: use audio_score instead of cnn_score ──────────────────────
    document.getElementById("audioScore").textContent =
      data.audio_score
        ? `${(data.audio_score.stressed * 100).toFixed(1)}% stressed`
        : "—";

    document.getElementById("nlpScore").textContent =
      data.nlp_score && data.nlp_score.stressed !== null
        ? `${(data.nlp_score.stressed * 100).toFixed(1)}% stressed`
        : "N/A";

    document.getElementById("methodLabel").textContent = data.method || "—";

    // ── Acoustic feature breakdown ─────────────────────────────────────────
    if (data.acoustic_features) {
      const af = data.acoustic_features;

      document.getElementById("featPitchMean").textContent =
        af.pitch_mean_hz + " Hz";

      document.getElementById("featPitchStd").textContent =
        af.pitch_std_hz + " Hz (σ)";

      document.getElementById("featRmsCv").textContent =
        (af.rms_variability * 100).toFixed(1) + "%";

      document.getElementById("featSpectral").textContent =
        af.spectral_centroid + " Hz";

      document.getElementById("featTempo").textContent =
        af.tempo_bpm + " BPM";

      document.getElementById("featMfcc").textContent =
        af.mfcc_variance;

      document.getElementById("acousticBreakdown").classList.remove("hidden");
    }

    voiceDetails.classList.remove("hidden");

  } catch (err) {
    voiceResult.textContent = "Network error: " + err.message;
    voiceResult.className   = "result-box show error";
  } finally {
    analyzeBtn.disabled = false;
  }
});