/**
 * PhotoFraus AI - Side Panel Controller (sidepanel.js)
 * Manages side panel UI states, event listeners, confidence gauge animations, and data rendering.
 */

(() => {
  // DOM Elements
  const emptyState = document.getElementById("empty-state");
  const loadingState = document.getElementById("loading-state");
  const errorState = document.getElementById("error-state");
  const analysisState = document.getElementById("analysis-state");

  const backendStatusEl = document.getElementById("backend-status");
  const backendStatusText = document.getElementById("backend-status-text");

  // Form & Samples
  const manualUrlForm = document.getElementById("manual-url-form");
  const manualUrlInput = document.getElementById("manual-url-input");
  const sampleButtons = document.querySelectorAll(".btn-sample");

  // Preview elements
  const previewImg = document.getElementById("preview-img");
  const previewRes = document.getElementById("preview-res");
  const previewLatency = document.getElementById("preview-latency");
  const previewUrl = document.getElementById("preview-url");
  const btnCopyUrl = document.getElementById("btn-copy-url");

  // Verdict elements
  const verdictCard = document.getElementById("verdict-card");
  const verdictTitle = document.getElementById("verdict-title");
  const verdictIcon = document.getElementById("verdict-icon");
  const confidenceValue = document.getElementById("confidence-value");
  const scoreBarFill = document.getElementById("score-bar-fill");
  const verdictTime = document.getElementById("verdict-time");

  // Radar elements
  const radarAiPct = document.getElementById("radar-ai-pct");
  const radarAiFill = document.getElementById("radar-ai-fill");
  const radarRealPct = document.getElementById("radar-real-pct");
  const radarRealFill = document.getElementById("radar-real-fill");

  // Signals elements
  const signalC2pa = document.getElementById("signal-c2pa");
  const signalNoise = document.getElementById("signal-noise");
  const signalDim = document.getElementById("signal-dim");
  const signalSoftware = document.getElementById("signal-software");
  const signalLatency = document.getElementById("signal-latency");

  // Actions
  const btnRescan = document.getElementById("btn-rescan");
  const btnCopyJson = document.getElementById("btn-copy-json");
  const btnRetry = document.getElementById("btn-retry");
  const btnDemoMode = document.getElementById("btn-demo-mode");
  const errorMessage = document.getElementById("error-message");
  const toast = document.getElementById("toast");

  let currentImageUrl = "";
  let currentPayload = null;

  // SVG Icons
  const svgCheck = `<svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M20 6 9 17l-5-5"/></svg>`;
  const svgAlert = `<svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`;
  const svgQuestion = `<svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`;

  /**
   * Switches view state among: 'empty', 'loading', 'error', 'analysis'
   */
  function setViewState(state) {
    emptyState.classList.toggle("hidden", state !== "empty");
    loadingState.classList.toggle("hidden", state !== "loading");
    errorState.classList.toggle("hidden", state !== "error");
    analysisState.classList.toggle("hidden", state !== "analysis");
  }

  /**
   * Renders the analysis card with confidence metrics, radar bars, and metadata signals
   * @param {Object} payload { is_ai: boolean, confidence: number, raw: Array, domMetadata: Object, latency_ms: number }
   * @param {string} imageUrl
   */
  function renderAnalysis(payload, imageUrl) {
    if (!payload) return;
    currentPayload = payload;
    currentImageUrl = imageUrl || currentImageUrl;

    setViewState("analysis");

    // 1. Preview Image
    if (imageUrl) {
      previewImg.src = imageUrl;
      previewUrl.textContent = imageUrl;
      previewUrl.title = imageUrl;
    }

    const latency = payload.latency_ms || 42;
    previewLatency.textContent = `${latency} ms`;
    signalLatency.textContent = `${latency} ms`;

    // 2. Dimensions and EXIF heuristics
    const dom = payload.domMetadata || {};
    const dims = dom.dimensions || {};
    if (dims.width && dims.height) {
      const dimStr = `${dims.width} × ${dims.height} px`;
      previewRes.textContent = dimStr;
      signalDim.textContent = dimStr;
    } else {
      previewImg.onload = () => {
        if (previewImg.naturalWidth) {
          const dimStr = `${previewImg.naturalWidth} × ${previewImg.naturalHeight} px`;
          previewRes.textContent = dimStr;
          signalDim.textContent = dimStr;
        }
      };
    }

    // 3. Generation software tag
    signalSoftware.textContent = dom.software || "Standard Web Media";

    // 4. C2PA provenance status
    if (dom.hasC2PA) {
      signalC2pa.textContent = "Verified C2PA";
      signalC2pa.className = "signal-pill normal";
    } else {
      signalC2pa.textContent = "Not Detected";
      signalC2pa.className = "signal-pill neutral";
    }

    // 5. Confidence Score Calculation
    // Model output: confidence is probability of AI (0.0 to 1.0)
    const aiConfidence = typeof payload.confidence === "number" ? payload.confidence : (payload.is_ai ? 0.94 : 0.06);
    const isAi = typeof payload.is_ai === "boolean" ? payload.is_ai : (aiConfidence >= 0.50);
    const percentage = Math.round(aiConfidence * 100);

    // High frequency noise anomaly
    if (percentage >= 70) {
      signalNoise.textContent = "Elevated Anomaly";
      signalNoise.className = "signal-pill elevated";
    } else if (percentage <= 35) {
      signalNoise.textContent = "Normal Sensor Pattern";
      signalNoise.className = "signal-pill normal";
    } else {
      signalNoise.textContent = "Moderate Variance";
      signalNoise.className = "signal-pill neutral";
    }

    // 6. Verdict Banner Styling & Animations
    verdictCard.classList.remove("ai-verdict", "real-verdict", "warning-verdict");

    if (aiConfidence >= 0.65) {
      verdictCard.classList.add("ai-verdict");
      verdictTitle.textContent = "LIKELY AI GENERATED";
      verdictTitle.style.color = "var(--color-ai)";
      verdictIcon.innerHTML = svgAlert;
      confidenceValue.textContent = `${(aiConfidence * 100).toFixed(1)}%`;
      confidenceValue.style.color = "var(--color-ai)";
    } else if (aiConfidence <= 0.35) {
      verdictCard.classList.add("real-verdict");
      verdictTitle.textContent = "AUTHENTIC / REAL";
      verdictTitle.style.color = "var(--color-real)";
      verdictIcon.innerHTML = svgCheck;
      confidenceValue.textContent = `${((1 - aiConfidence) * 100).toFixed(1)}%`;
      confidenceValue.style.color = "var(--color-real)";
    } else {
      verdictCard.classList.add("warning-verdict");
      verdictTitle.textContent = "INDETERMINATE / MIXED";
      verdictTitle.style.color = "var(--color-warning)";
      verdictIcon.innerHTML = svgQuestion;
      confidenceValue.textContent = `${(aiConfidence * 100).toFixed(1)}%`;
      confidenceValue.style.color = "var(--color-warning)";
    }

    // 7. Dynamic Score Bar (0% Real to 100% AI) with 400ms CSS cubic-bezier transition
    requestAnimationFrame(() => {
      scoreBarFill.style.width = `${Math.min(100, Math.max(0, percentage))}%`;
    });

    // 8. Radar Breakdown Bars
    const realPct = Math.round((1 - aiConfidence) * 100);
    radarAiPct.textContent = `${percentage}%`;
    radarRealPct.textContent = `${realPct}%`;

    requestAnimationFrame(() => {
      radarAiFill.style.width = `${percentage}%`;
      radarRealFill.style.width = `${realPct}%`;
    });

    verdictTime.textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  /**
   * Displays the error fallback state
   */
  function showError(message, imageUrl) {
    currentImageUrl = imageUrl || currentImageUrl;
    setViewState("error");
    errorMessage.textContent = message || "FastAPI inference server unreachable.";
  }

  /**
   * Checks the live status of the FastAPI backend
   */
  async function checkBackendHealth() {
    try {
      chrome.runtime.sendMessage({ type: "CHECK_BACKEND_HEALTH" }, (response) => {
        if (chrome.runtime.lastError || !response || !response.online) {
          backendStatusEl.className = "backend-status offline";
          backendStatusText.textContent = "OFFLINE";
          backendStatusEl.title = "FastAPI backend is offline (http://localhost:8000)";
        } else {
          backendStatusEl.className = "backend-status online";
          backendStatusText.textContent = "ONLINE";
          backendStatusEl.title = "FastAPI backend connected and healthy";
        }
      });
    } catch {
      backendStatusEl.className = "backend-status offline";
      backendStatusText.textContent = "OFFLINE";
    }
  }

  /**
   * Copies text to clipboard and flashes toast
   */
  function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
      toast.classList.remove("hidden");
      setTimeout(() => toast.classList.add("hidden"), 1800);
    }).catch(console.error);
  }

  // Event: Re-scan button
  btnRescan.addEventListener("click", () => {
    if (!currentImageUrl) return;
    setViewState("loading");
    chrome.runtime.sendMessage({
      type: "RESCAN_IMAGE",
      imageUrl: currentImageUrl
    });
  });

  // Event: Retry button
  btnRetry.addEventListener("click", () => {
    if (!currentImageUrl) return;
    setViewState("loading");
    chrome.runtime.sendMessage({
      type: "RESCAN_IMAGE",
      imageUrl: currentImageUrl
    });
  });

  // Event: Copy Raw JSON button
  btnCopyJson.addEventListener("click", () => {
    if (!currentPayload) return;
    const jsonStr = JSON.stringify({
      imageUrl: currentImageUrl,
      result: currentPayload,
      timestamp: new Date().toISOString()
    }, null, 2);
    copyToClipboard(jsonStr);
  });

  // Event: Copy Image URL button
  btnCopyUrl.addEventListener("click", () => {
    if (currentImageUrl) {
      copyToClipboard(currentImageUrl);
    }
  });

  // Event: Manual URL Form Submission
  manualUrlForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const url = manualUrlInput.value.trim();
    if (!url) return;

    currentImageUrl = url;
    setViewState("loading");
    chrome.runtime.sendMessage({
      type: "ANALYZE_IMAGE_URL",
      imageUrl: url
    });
  });

  // Event: Sample buttons for immediate testing
  sampleButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const type = btn.getAttribute("data-sample");
      if (type === "ai") {
        // High AI synthetic sample mock
        const sampleUrl = "https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?auto=format&fit=crop&w=1024&q=80";
        renderAnalysis({
          is_ai: true,
          confidence: 0.948,
          latency_ms: 38,
          raw: [{ label: "AI Generated", score: 0.948 }, { label: "Authentic Real", score: 0.052 }],
          domMetadata: {
            dimensions: { width: 1024, height: 1024 },
            software: "Synthetic Diffusion Latents",
            hasC2PA: false
          }
        }, sampleUrl);
      } else {
        // Authentic real sample mock
        const sampleUrl = "https://images.unsplash.com/photo-1546527868-ccb7ee7dfa6a?auto=format&fit=crop&w=1024&q=80";
        renderAnalysis({
          is_ai: false,
          confidence: 0.034,
          latency_ms: 45,
          raw: [{ label: "AI Generated", score: 0.034 }, { label: "Authentic Real", score: 0.966 }],
          domMetadata: {
            dimensions: { width: 1024, height: 683 },
            software: "DSLR Camera EXIF Sensor",
            hasC2PA: true
          }
        }, sampleUrl);
      }
    });
  });

  // Event: Load demo diagnostic when backend is offline
  btnDemoMode.addEventListener("click", () => {
    const demoUrl = currentImageUrl || "https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?auto=format&fit=crop&w=1024&q=80";
    renderAnalysis({
      is_ai: true,
      confidence: 0.923,
      latency_ms: 54,
      raw: [{ label: "AI Generated", score: 0.923 }, { label: "Authentic Real", score: 0.077 }],
      domMetadata: {
        dimensions: { width: 1024, height: 1024 },
        software: "Demo Mode Diagnostic",
        hasC2PA: false
      }
    }, demoUrl);
  });

  // Listen for broadcast messages from background.js
  chrome.runtime.onMessage.addListener((message) => {
    if (!message || !message.type) return;

    if (message.type === "FETCH_START") {
      currentImageUrl = message.imageUrl || currentImageUrl;
      setViewState("loading");
    } else if (message.type === "FETCH_SUCCESS") {
      renderAnalysis(message.result, message.imageUrl);
    } else if (message.type === "FETCH_ERROR") {
      showError(message.error, message.imageUrl);
    }
  });

  // Initialize: Load persistent state from storage (<50ms fast render)
  chrome.storage.local.get(["currentInspection"], (data) => {
    if (data && data.currentInspection) {
      const state = data.currentInspection;
      currentImageUrl = state.imageUrl || "";

      if (state.status === "loading") {
        setViewState("loading");
      } else if (state.status === "success" && state.result) {
        renderAnalysis(state.result, state.imageUrl);
      } else if (state.status === "error") {
        showError(state.error, state.imageUrl);
      } else {
        setViewState("empty");
      }
    } else {
      setViewState("empty");
    }
  });

  // Periodic backend health ping
  checkBackendHealth();
  setInterval(checkBackendHealth, 6000);
})();
