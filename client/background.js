/**
 * PhotoFraus AI - Service Worker (background.js)
 * Manages context menus, side panel coordination, API inference proxying, and state persistence.
 */

const BACKEND_URLS = ["http://127.0.0.1:8000", "http://localhost:8000"];

async function fetchFromBackend(endpoint, options = {}) {
  let lastError = null;
  for (const baseUrl of BACKEND_URLS) {
    try {
      const response = await fetch(`${baseUrl}${endpoint}`, options);
      return response;
    } catch (err) {
      lastError = err;
    }
  }
  throw lastError || new Error("Failed to connect to backend server");
}

/**
 * Registers right-click context menu on image elements
 */
function initContextMenus() {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: "inspect-image-ai",
      title: "Inspect Image with PhotoFraus AI",
      contexts: ["image"]
    });
  });
}

// Extension installation lifecycle
chrome.runtime.onInstalled.addListener(() => {
  initContextMenus();

  // Configure action icon click to open side panel automatically
  if (chrome.sidePanel && chrome.sidePanel.setPanelBehavior) {
    chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch((err) => {
      console.warn("Could not set panel behavior:", err);
    });
  }
});

// Context menu click listener
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "inspect-image-ai") {
    handleImageInspection(info, tab);
  }
});

/**
 * Handles image inspection workflow triggered by context menu
 * @param {chrome.contextMenus.OnClickData} info
 * @param {chrome.tabs.Tab} tab
 */
async function handleImageInspection(info, tab) {
  const imageUrl = info.srcUrl;
  if (!imageUrl) return;

  const tabId = tab ? tab.id : undefined;

  // 1. Open side panel immediately in response to user gesture
  if (tabId && chrome.sidePanel && chrome.sidePanel.open) {
    try {
      await chrome.sidePanel.open({ tabId });
    } catch (e) {
      console.warn("Failed to open sidePanel by tabId, trying windowId:", e);
      if (tab.windowId) {
        chrome.sidePanel.open({ windowId: tab.windowId }).catch(console.error);
      }
    }
  }

  // 2. Perform the inspection
  await performInspection(imageUrl, tabId);
}

/**
 * Dispatches prediction request to FastAPI and synchronizes state with side panel and content script
 * @param {string} imageUrl
 * @param {number|undefined} tabId
 */
async function performInspection(imageUrl, tabId) {
  const startTime = Date.now();

  // 1. Set immediate loading state in storage for <50ms UI feedback
  const loadingState = {
    imageUrl,
    tabId,
    status: "loading",
    timestamp: startTime
  };

  await chrome.storage.local.set({ currentInspection: loadingState });

  // 2. Dispatch FETCH_START message to active views
  chrome.runtime.sendMessage({
    type: "FETCH_START",
    imageUrl,
    timestamp: startTime
  }).catch(() => {});

  // 3. Notify content script to inject scanning badge and extract DOM metadata
  let domMetadata = null;
  if (tabId) {
    try {
      const response = await chrome.tabs.sendMessage(tabId, {
        type: "INSPECT_IMAGE_START",
        imageUrl
      });
      if (response && response.metadata) {
        domMetadata = response.metadata;
      }
    } catch {
      // Content script may not be loaded on internal chrome:// pages or restricted frames
    }
  }

  // 4. Send image_url to FastAPI backend (POST /predict)
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 7000); // 7s timeout

    const response = await fetchFromBackend("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json"
      },
      body: JSON.stringify({ image_url: imageUrl }),
      signal: controller.signal
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      let errDetail = `HTTP ${response.status} (${response.statusText})`;
      try {
        const errorJson = await response.json();
        if (errorJson.detail) errDetail = errorJson.detail;
      } catch {}

      if (response.status === 403) {
        errDetail = "403 Host Restricted: Target server blocked the image stream.";
      } else if (response.status === 415 || response.status === 422) {
        errDetail = "Unsupported Image Format: Target image must be JPEG, PNG, or WEBP.";
      }

      throw new Error(errDetail);
    }

    const data = await response.json();
    const latencyMs = data.latency_ms || (Date.now() - startTime);

    const resultPayload = {
      is_ai: typeof data.is_ai === "boolean" ? data.is_ai : (data.confidence >= 0.5),
      confidence: typeof data.confidence === "number" ? data.confidence : 0.5,
      latency_ms: Math.round(latencyMs),
      raw: data.raw || [
        { label: "AI Generated", score: data.confidence },
        { label: "Authentic Real", score: 1 - data.confidence }
      ],
      domMetadata: domMetadata || data.metadata || null
    };

    const successState = {
      imageUrl,
      tabId,
      status: "success",
      result: resultPayload,
      timestamp: Date.now()
    };

    // Save state
    await chrome.storage.local.set({ currentInspection: successState });

    // Notify side panel
    chrome.runtime.sendMessage({
      type: "FETCH_SUCCESS",
      imageUrl,
      result: resultPayload
    }).catch(() => {});

    // Notify content script badge
    if (tabId) {
      chrome.tabs.sendMessage(tabId, {
        type: "INSPECT_IMAGE_RESULT",
        imageUrl,
        isAi: resultPayload.is_ai,
        score: resultPayload.confidence,
        status: "complete"
      }).catch(() => {});
    }

  } catch (error) {
    let humanReadableError = error.message;

    if (error.name === "AbortError") {
      humanReadableError = "Inference Timed Out (>7s): Image stream download or model pass exceeded threshold.";
    } else if (error.message.includes("Failed to fetch") || error.message.includes("NetworkError")) {
      humanReadableError = "Backend Offline: FastAPI server not reachable at http://localhost:8000.";
    }

    const errorState = {
      imageUrl,
      tabId,
      status: "error",
      error: humanReadableError,
      domMetadata,
      timestamp: Date.now()
    };

    // Save error state
    await chrome.storage.local.set({ currentInspection: errorState });

    // Notify side panel
    chrome.runtime.sendMessage({
      type: "FETCH_ERROR",
      imageUrl,
      error: humanReadableError,
      domMetadata
    }).catch(() => {});

    // Notify content script badge
    if (tabId) {
      chrome.tabs.sendMessage(tabId, {
        type: "INSPECT_IMAGE_RESULT",
        imageUrl,
        status: "error",
        errorMessage: humanReadableError
      }).catch(() => {});
    }
  }
}

// Runtime message dispatcher
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message || !message.type) return;

  if (message.type === "RESCAN_IMAGE" || message.type === "ANALYZE_IMAGE_URL") {
    const tabId = sender.tab ? sender.tab.id : undefined;
    performInspection(message.imageUrl, tabId);
    sendResponse({ success: true });
    return true;
  }

  if (message.type === "CHECK_BACKEND_HEALTH") {
    fetchFromBackend("/health", { signal: AbortSignal.timeout(2000) })
      .then((res) => res.ok)
      .then((online) => sendResponse({ online }))
      .catch(() => sendResponse({ online: false }));
    return true;
  }

  if (message.type === "OPEN_SIDEPANEL_REQUEST") {
    const tabId = sender.tab ? sender.tab.id : undefined;
    if (tabId && chrome.sidePanel && chrome.sidePanel.open) {
      chrome.sidePanel.open({ tabId }).catch(console.error);
    }
    if (message.imageUrl) {
      performInspection(message.imageUrl, tabId);
    }
    sendResponse({ success: true });
    return true;
  }
});
