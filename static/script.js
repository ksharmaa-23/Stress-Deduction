document.getElementById("predictBtn").addEventListener("click", async () => {
  const snoring = document.getElementById("snoring").value;
  const temp = document.getElementById("temp").value;
  const hours = document.getElementById("hours").value;
  const hr = document.getElementById("hr").value;

  const payload = {
    snoring_range: snoring, 
    body_temperature: temp,     
     hours_sleep: hours,   
   heart_rate: hr
  };

  const resultEl = document.getElementById("result");
  resultEl.textContent = "Predicting...";

  try {
    const res = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const err = await res.json();
      resultEl.textContent = err.error || "Prediction failed.";
      return;
    }

    const data = await res.json();
    resultEl.textContent = `${data.message} (confidence: ${data.probability})`;
  } catch (err) {
    resultEl.textContent = "Network error: " + err.message;
  }
});
