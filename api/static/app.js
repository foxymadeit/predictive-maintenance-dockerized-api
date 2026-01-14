async function handleResponse(res) {
  if (res.ok) {
    const contentType = res.headers.get("Content-Type");
    if (contentType && contentType.includes("application/json")) {
      return await res.json();
    }
    return await res.blob();
  }

  let errorMessage = `${res.status} ${res.statusText}`;
  try {
    const errorDetail = await res.json();
    if (errorDetail.detail) {
      if (Array.isArray(errorDetail.detail)) {
        errorMessage = errorDetail.detail
          .map(err => `• ${err.loc.join(" ➔ ")}: ${err.msg}`)
          .join("\n");
      } else {
        errorMessage = errorDetail.detail;
      }
    }
  } catch (e) {
    const text = await res.text();
    if (text) errorMessage += `: ${text}`;
  }
  
  throw new Error(errorMessage);
}


function setStatus(msg) {
  document.getElementById("status").textContent = msg || "";
}

function readForm() {
  return {
    "Air temperature [K]": Number(document.getElementById("air").value),
    "Process temperature [K]": Number(document.getElementById("proc").value),
    "Rotational speed [rpm]": Number(document.getElementById("rpm").value),
    "Torque [Nm]": Number(document.getElementById("torque").value),
    "Tool wear [min]": Number(document.getElementById("wear").value),
    "Type": document.getElementById("type").value
  };
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });

  return await handleResponse(res)
}

async function postForBlob(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });

  return await handleResponse(res);
}

function pretty(obj) {
  return JSON.stringify(obj, null, 2);
}

document.getElementById("btnPredict").addEventListener("click", async () => {
  try {
    setStatus("Predicting…");
    const payload = readForm();
    const data = await postJSON("/predict", payload);
    document.getElementById("out").textContent = pretty(data);
    setStatus("OK");
  } catch (e) {
    setStatus("Error");
    document.getElementById("out").textContent = String(e);
  }
});

document.getElementById("btnExplain").addEventListener("click", async () => {
  try {
    setStatus("Explaining…");
    const payload = readForm();
    const data = await postJSON("/explain?top_k=8", payload);
    document.getElementById("out").textContent = pretty(data);
    setStatus("OK");
  } catch (e) {
    setStatus("Error");
    document.getElementById("out").textContent = String(e);
  }
});

document.getElementById("btnPlot").addEventListener("click", async () => {
  try {
    setStatus("Rendering plot…");
    const payload = readForm();
    const blob = await postForBlob("/explain/plot", payload);
    if (document.getElementById("plot").src.startsWith("blob:")) {
        URL.revokeObjectURL(document.getElementById("plot").src);
    }
    const url = URL.createObjectURL(blob);
    document.getElementById("plot").src = url;
    setStatus("OK");
  } catch (e) {
    setStatus("Error");
    document.getElementById("out").textContent = String(e);
  }
});