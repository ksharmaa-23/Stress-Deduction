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
    snoring_range: Number(snoring),
    respiration_rate : Number(document.getElementById("rr")?.value || 15),
    body_temperature : Number(temp),
    limb_movement    : Number(document.getElementById("lm")?.value || 8),
    blood_oxygen     : Number(document.getElementById("bo")?.value || 97),
    eye_movement     : Number(document.getElementById("rem")?.value || 12),
    hours_sleep      : Number(hours),
    heart_rate       : Number(hr,
    body_temperature : Number(temp),
    hours_sleep      : Number(hours),
    heart_rate       : Number(hr)
  };

  const resultEl = document.getElementById("result");
  resultEl.textContent = "Analyzing...";
  resultEl.className   = "result-box show";

  try {
    const res = await fetch("/predict", {
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
    // Convert webm to WAV using AudioContext
    const arrayBuffer = await recordedBlob.arrayBuffer();
    const audioCtx    = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    const wavBlob     = audioBufferToWav(audioBuffer);

    const formData = new FormData();
    formData.append("audio",      wavBlob, "recording.wav");
    formData.append("transcript", transcriptText.value.trim());

    const res = await fetch("/predict-voice", { method: "POST", body: formData });

    if (!res.ok) {
      let errMsg = "Voice analysis failed.";
      try {
        const err = await res.json();
        errMsg = res.status === 503
          ? "⚠️ Voice model not available. Use the Physiological tab."
          : (err.error || errMsg);
      } catch(_) {}
      voiceResult.textContent = errMsg;
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

    const audioScoreEl = document.getElementById("audioScore");
    if (audioScoreEl) {
      audioScoreEl.textContent = data.audio_score
        ? `${(data.audio_score.stressed * 100).toFixed(1)}% stressed`
        : "—";
    }

    const nlpScoreEl = document.getElementById("nlpScore");
    if (nlpScoreEl) {
      nlpScoreEl.textContent = data.nlp_score && data.nlp_score.stressed !== null
        ? `${(data.nlp_score.stressed * 100).toFixed(1)}% stressed`
        : "N/A";
    }

    const methodEl = document.getElementById("methodLabel");
    if (methodEl) methodEl.textContent = data.method || "—";

    // Acoustic features
    if (data.acoustic_features) {
      const af = data.acoustic_features;
      const set = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
      };
      set("featPitchMean",  (af.pitch_mean_hz || 0) + " Hz");
      set("featPitchStd",   (af.pitch_std_hz  || 0) + " Hz");
      set("featRmsCv",      ((af.rms_variability || 0) * 100).toFixed(1) + "%");
      set("featSpectral",   (af.spectral_centroid || 0) + " Hz");
      set("featTempo",      "—");
      set("featMfcc",       (af.mfcc_variance || 0).toFixed(2));

      const breakdown = document.getElementById("acousticBreakdown");
      if (breakdown) breakdown.classList.remove("hidden");
    }

    voiceDetails.classList.remove("hidden");

  } catch (err) {
    voiceResult.textContent = "Network error: " + err.message;
    voiceResult.className   = "result-box show error";
  } finally {
    analyzeBtn.disabled = false;
  }
});


// ── Convert AudioBuffer → WAV blob ───────────────────────────────────────────
function audioBufferToWav(buffer) {
  const numChannels = 1;
  const sampleRate  = buffer.sampleRate;
  const samples     = buffer.getChannelData(0);
  const dataLength  = samples.length * 2;
  const arrayBuf    = new ArrayBuffer(44 + dataLength);
  const view        = new DataView(arrayBuf);

  const writeStr = (off, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i));
  };

  writeStr(0, 'RIFF');
  view.setUint32(4,  36 + dataLength, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1,  true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate,  true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2,  true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, dataLength, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    offset += 2;
  }

  return new Blob([arrayBuf], { type: "audio/wav" });
}
