/**
 * PhotoFraus AI - Content Script
 * Handles DOM image tracking, client-side metadata extraction, and isolated Shadow DOM overlay badges.
 */

(() => {
  // Prevent duplicate injections
  if (window.__photofraus_injected__) return;
  window.__photofraus_injected__ = true;

  let lastClickedImage = null;
  const activeBadges = new Map(); // img element -> shadow host element

  // Track the most recent right-clicked image element
  document.addEventListener(
    "contextmenu",
    (e) => {
      const target = e.target;
      if (target instanceof HTMLImageElement) {
        lastClickedImage = target;
      } else if (target instanceof HTMLElement) {
        // Check if there is an image inside or if the target has a background image
        const img = target.querySelector("img") || target.closest("img");
        if (img) {
          lastClickedImage = img;
        } else {
          // Check computed background-image
          const bg = window.getComputedStyle(target).backgroundImage;
          if (bg && bg.startsWith('url("') && bg !== 'url("none")') {
            lastClickedImage = target;
          }
        }
      }
    },
    true
  );

  /**
   * Fast client-side scan for native EXIF, prompt hints, and generation metadata.
   * @param {HTMLImageElement|HTMLElement} imageElement
   * @returns {Object} MetadataResult
   */
  function extractImageMetadata(imageElement) {
    if (!imageElement) {
      return { hasExif: false, software: "Unknown", promptHint: null, dimensions: { width: 0, height: 0 } };
    }

    const width = imageElement.naturalWidth || imageElement.clientWidth || 0;
    const height = imageElement.naturalHeight || imageElement.clientHeight || 0;
    const src = imageElement.currentSrc || imageElement.src || "";
    const alt = (imageElement.alt || "").toLowerCase();
    const title = (imageElement.title || "").toLowerCase();

    let software = "Standard Web Media";
    let promptHint = null;
    let hasExif = false;

    // Fast heuristic scan for common generative model tokens in metadata/attributes/URLs
    const aiKeywords = [
      { key: "midjourney", name: "Midjourney" },
      { key: "dall-e", name: "OpenAI DALL-E" },
      { key: "dalle", name: "OpenAI DALL-E" },
      { key: "stable-diffusion", name: "Stable Diffusion" },
      { key: "stablediffusion", name: "Stable Diffusion" },
      { key: "comfyui", name: "ComfyUI" },
      { key: "novelai", name: "NovelAI" },
      { key: "firefly", name: "Adobe Firefly" },
      { key: "flux", name: "Flux AI" },
      { key: "civitai", name: "Civitai Gen" }
    ];

    const haystack = `${src} ${alt} ${title}`.toLowerCase();
    for (const item of aiKeywords) {
      if (haystack.includes(item.key)) {
        software = item.name;
        promptHint = alt || title || "AI Generation Metadata detected in image attributes";
        break;
      }
    }

    // Check for common generative dimensions (1024x1024, 512x512, 768x768, 896x1152)
    const isSquareGenRatio = (width === 1024 && height === 1024) || (width === 512 && height === 512) || (width === 768 && height === 768);

    return {
      hasExif,
      software,
      promptHint,
      isSquareGenRatio,
      dimensions: { width, height },
      srcUrl: src
    };
  }

  /**
   * Finds an image element on page that matches the provided URL
   * @param {string} url
   * @returns {HTMLImageElement|null}
   */
  function findImageByUrl(url) {
    if (!url) return lastClickedImage;
    if (lastClickedImage && (lastClickedImage.currentSrc === url || lastClickedImage.src === url)) {
      return lastClickedImage;
    }

    const images = Array.from(document.querySelectorAll("img"));
    for (const img of images) {
      if (img.currentSrc === url || img.src === url) {
        return img;
      }
    }
    return lastClickedImage;
  }

  /**
   * Mounts an isolated Shadow DOM badge directly on the inspected image
   * @param {HTMLImageElement|HTMLElement} imageElement
   * @param {number|null} score (0.00 - 1.00)
   * @param {boolean|null} isAi
   * @param {'scanning'|'complete'|'error'} status
   * @param {string} message
   */
  function injectOverlayBadge(imageElement, score = null, isAi = null, status = "complete", message = "") {
    if (!imageElement || !(imageElement instanceof Element)) return;

    // Remove existing badge on this image if present
    if (activeBadges.has(imageElement)) {
      const existing = activeBadges.get(imageElement);
      if (existing && existing.parentNode) {
        existing.parentNode.removeChild(existing);
      }
      activeBadges.delete(imageElement);
    }

    // Create shadow host container
    const host = document.createElement("div");
    host.setAttribute("data-photofraus-badge", "true");
    host.style.position = "absolute";
    host.style.zIndex = "2147483647";
    host.style.pointerEvents = "auto";
    host.style.margin = "0";
    host.style.padding = "0";

    const shadow = host.attachShadow({ mode: "open" });

    // Load external stylesheet & embed fallback styles inside shadow root for zero DOM pollution
    const style = document.createElement("style");
    style.textContent = `
      :host {
        all: initial;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        user-select: none;
      }
      .pf-badge-wrapper {
        position: relative;
        display: inline-flex;
        align-items: center;
      }
      .pf-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        height: 28px;
        padding: 0 10px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #F0F6FC;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        cursor: pointer;
        transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.2s ease;
      }
      .pf-badge:hover {
        transform: translateY(-1px) scale(1.04);
      }
      .pf-badge.scanning {
        background: rgba(22, 27, 34, 0.92);
        border: 1px solid rgba(0, 240, 255, 0.5);
        color: #00F0FF;
        animation: pf-pulse 1.6s ease-in-out infinite;
      }
      .pf-spinner {
        width: 12px;
        height: 12px;
        border: 2px solid rgba(0, 240, 255, 0.3);
        border-top-color: #00F0FF;
        border-radius: 50%;
        animation: pf-spin 0.8s linear infinite;
      }
      .pf-badge.ai {
        background: #F85149;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #FFFFFF;
        box-shadow: 0 4px 14px rgba(248, 81, 73, 0.45);
      }
      .pf-badge.real {
        background: #2EA043;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #FFFFFF;
        box-shadow: 0 4px 14px rgba(46, 160, 67, 0.45);
      }
      .pf-badge.warning {
        background: #D29922;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #FFFFFF;
      }
      .pf-tooltip {
        position: absolute;
        top: calc(100% + 6px);
        right: 0;
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 11px;
        color: #F0F6FC;
        white-space: nowrap;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
        opacity: 0;
        visibility: hidden;
        transform: translateY(-4px);
        transition: opacity 0.15s ease, transform 0.15s ease, visibility 0.15s;
        pointer-events: none;
        z-index: 10;
      }
      .pf-badge-wrapper:hover .pf-tooltip {
        opacity: 1;
        visibility: visible;
        transform: translateY(0);
      }
      @keyframes pf-pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(0, 240, 255, 0.4); }
        50% { box-shadow: 0 0 12px 2px rgba(0, 240, 255, 0.7); }
      }
      @keyframes pf-spin {
        to { transform: rotate(360deg); }
      }
    `;

    // SVG Icons
    const checkIcon = `<svg viewBox="0 0 16 16" width="14" height="14" fill="currentColor"><path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path></svg>`;
    const alertIcon = `<svg viewBox="0 0 16 16" width="14" height="14" fill="currentColor"><path d="M6.457 1.047c.659-1.234 2.427-1.234 3.086 0l6.082 11.378A1.75 1.75 0 0 1 14.082 15H1.918a1.75 1.75 0 0 1-1.543-2.575Zm1.763.707a.25.25 0 0 0-.44 0L1.698 13.132a.25.25 0 0 0 .22.368h12.164a.25.25 0 0 0 .22-.368Zm.53 3.996v2.5a.75.75 0 0 1-1.5 0v-2.5a.75.75 0 0 1 1.5 0ZM9 11a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z"></path></svg>`;
    const infoIcon = `<svg viewBox="0 0 16 16" width="14" height="14" fill="currentColor"><path d="M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8Zm8-6.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13ZM6.5 7.75A.75.75 0 0 1 7.25 7h1a.75.75 0 0 1 .75.75v2.75h.25a.75.75 0 0 1 0 1.5h-2a.75.75 0 0 1 0-1.5h.25v-2h-.25a.75.75 0 0 1-.75-.75ZM8 6a1 1 0 1 1 0-2 1 1 0 0 1 0 2Z"></path></svg>`;

    let badgeClass = "scanning";
    let icon = `<div class="pf-spinner"></div>`;
    let label = "SCANNING...";
    let tooltipText = "PhotoFraus AI analyzing image...";

    if (status === "scanning") {
      badgeClass = "scanning";
      icon = `<div class="pf-spinner"></div>`;
      label = "SCANNING...";
      tooltipText = "Analyzing neural visual patterns...";
    } else if (status === "error") {
      badgeClass = "warning";
      icon = infoIcon;
      label = "OFFLINE";
      tooltipText = message || "Backend inference offline";
    } else if (isAi === true) {
      const pct = score !== null ? Math.round(score * 100) : 90;
      badgeClass = "ai";
      icon = alertIcon;
      label = `AI ${pct}%`;
      tooltipText = `Confidence: ${(score * 100).toFixed(1)}% AI Generated`;
    } else if (isAi === false) {
      const pct = score !== null ? Math.round((1 - score) * 100) : 95;
      badgeClass = "real";
      icon = checkIcon;
      label = `REAL ${pct}%`;
      tooltipText = `Confidence: ${((1 - score) * 100).toFixed(1)}% Authentic Image`;
    }

    const wrapper = document.createElement("div");
    wrapper.className = "pf-badge-wrapper";
    wrapper.innerHTML = `
      <div class="pf-badge ${badgeClass}" id="badge-pill">
        ${icon}
        <span>${label}</span>
      </div>
      <div class="pf-tooltip">
        <div><strong>PhotoFraus AI</strong></div>
        <div>${tooltipText}</div>
        <div style="color: #00F0FF; margin-top: 4px; font-size: 10px;">Click to view side panel details</div>
      </div>
    `;

    // Click handler to open side panel
    wrapper.querySelector("#badge-pill").addEventListener("click", (e) => {
      e.stopPropagation();
      chrome.runtime.sendMessage({
        type: "OPEN_SIDEPANEL_REQUEST",
        imageUrl: imageElement.currentSrc || imageElement.src
      });
    });

    shadow.appendChild(style);
    shadow.appendChild(wrapper);

    // Position badge over top-right corner of image
    function updatePosition() {
      if (!imageElement.isConnected) {
        if (host.parentNode) host.parentNode.removeChild(host);
        window.removeEventListener("resize", updatePosition);
        window.removeEventListener("scroll", updatePosition);
        activeBadges.delete(imageElement);
        return;
      }
      const rect = imageElement.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return;

      const top = rect.top + window.scrollY + 8;
      const left = rect.left + window.scrollX + rect.width - 105;

      host.style.top = `${Math.max(window.scrollY + 4, top)}px`;
      host.style.left = `${Math.max(window.scrollX + 4, left)}px`;
    }

    document.body.appendChild(host);
    activeBadges.set(imageElement, host);

    updatePosition();
    window.addEventListener("resize", updatePosition, { passive: true });
    window.addEventListener("scroll", updatePosition, { passive: true });

    // Auto update on image load in case dimensions changed
    imageElement.addEventListener("load", updatePosition, { once: true });
  }

  // Listen for messages from background script
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (!message || !message.type) return;

    if (message.type === "INSPECT_IMAGE_START") {
      const targetImg = findImageByUrl(message.imageUrl);
      if (targetImg) {
        injectOverlayBadge(targetImg, null, null, "scanning");
        const metadata = extractImageMetadata(targetImg);
        sendResponse({ success: true, metadata });
      } else {
        sendResponse({ success: false, reason: "Image not found on DOM" });
      }
      return true;
    }

    if (message.type === "INSPECT_IMAGE_RESULT") {
      const targetImg = findImageByUrl(message.imageUrl);
      if (targetImg) {
        injectOverlayBadge(
          targetImg,
          message.score,
          message.isAi,
          message.status || "complete",
          message.errorMessage || ""
        );
      }
      sendResponse({ success: true });
      return true;
    }

    if (message.type === "GET_IMAGE_METADATA") {
      const targetImg = findImageByUrl(message.imageUrl);
      const metadata = extractImageMetadata(targetImg);
      sendResponse({ metadata });
      return true;
    }
  });
})();
