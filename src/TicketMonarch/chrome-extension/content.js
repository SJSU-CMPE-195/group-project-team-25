// TicketMonarch Telemetry Collector — Content Script
// Injected into every page. Captures mouse, clicks, keystrokes, scroll,
// client hints, and network metadata. Segments data around idle gaps
// so training data stays temporally coherent.

(() => {
  'use strict'

  // ── Configuration ──────────────────────────────────────────────────
  const MOUSE_SAMPLE_INTERVAL_MS = 15   // ~66 Hz sampling
  const FLUSH_INTERVAL_MS = 5000        // send buffers every 5s
  const IDLE_THRESHOLD_MS = 3000        // 3s inactivity → end segment
  const SCROLL_THROTTLE_MS = 50         // ~20 Hz scroll sampling

  // Non-sensitive special keys worth logging for behavioral analysis.
  // Letters, digits, and modifiers (Shift, Ctrl, Alt, Meta) are excluded.
  const LOGGABLE_KEYS = new Set([
    'Backspace', 'Delete', 'Tab', 'Enter', 'Escape',
    'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight',
    'Home', 'End', 'PageUp', 'PageDown',
    'Insert', 'CapsLock', 'NumLock', 'ScrollLock',
    'ContextMenu', 'PrintScreen', 'Pause',
    'F1', 'F2', 'F3', 'F4', 'F5', 'F6',
    'F7', 'F8', 'F9', 'F10', 'F11', 'F12'
  ])

  // ── State ──────────────────────────────────────────────────────────
  let recording = false
  let sessionId = null
  let segmentId = 0
  let segmentStartTime = null

  let mouseBuffer = []
  let clickBuffer = []
  let keystrokeBuffer = []
  let scrollBuffer = []

  let lastMouseEvent = null
  let lastInteractionTime = performance.now()
  let lastClickTimestamp = null
  let lastKeyTimestampByField = {}
  let lastScrollTimestamp = null
  let lastScrollX = window.scrollX
  let lastScrollY = window.scrollY

  let mouseIntervalId = null
  let flushIntervalId = null
  let idleCheckIntervalId = null
  let isIdle = false
  let pageVisible = true

  // ── Client Hints (collected once) ─────────────────────────────────
  function collectClientHints() {
    const hints = {
      screenWidth: screen.width,
      screenHeight: screen.height,
      innerWidth: window.innerWidth,
      innerHeight: window.innerHeight,
      devicePixelRatio: window.devicePixelRatio || 1,
      language: navigator.language,
      languages: navigator.languages ? [...navigator.languages] : [navigator.language],
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      timezoneOffset: new Date().getTimezoneOffset(),
      hardwareConcurrency: navigator.hardwareConcurrency || null,
      deviceMemory: navigator.deviceMemory || null,
      platform: navigator.platform,
      userAgent: navigator.userAgent,
      maxTouchPoints: navigator.maxTouchPoints || 0,
      colorDepth: screen.colorDepth
    }
    return hints
  }

  // ── Network Metadata (collected once) ─────────────────────────────
  function collectNetworkMeta() {
    const meta = {}

    // Network Information API
    const conn = navigator.connection || navigator.mozConnection || navigator.webkitConnection
    if (conn) {
      meta.effectiveType = conn.effectiveType || null
      meta.downlink = conn.downlink || null
      meta.rtt = conn.rtt || null
      meta.saveData = conn.saveData || false
    }

    // Navigation timing for page load RTT
    try {
      const navEntries = performance.getEntriesByType('navigation')
      if (navEntries.length > 0) {
        const nav = navEntries[0]
        meta.pageLoadDuration = nav.loadEventEnd - nav.startTime
        meta.domContentLoaded = nav.domContentLoadedEventEnd - nav.startTime
        meta.ttfb = nav.responseStart - nav.requestStart
        meta.dnsLookup = nav.domainLookupEnd - nav.domainLookupStart
        meta.tcpConnect = nav.connectEnd - nav.connectStart
      }
    } catch {
      // Navigation Timing not available
    }

    return meta
  }

  // ── Interaction timestamp updater ─────────────────────────────────
  function touchInteraction() {
    lastInteractionTime = performance.now()

    // If we were idle, start a new segment
    if (isIdle) {
      isIdle = false
      startNewSegment()
    }
  }

  // ── Segment management ────────────────────────────────────────────
  function startNewSegment() {
    segmentId++
    segmentStartTime = Date.now()
    mouseBuffer = []
    clickBuffer = []
    keystrokeBuffer = []
    scrollBuffer = []
    lastClickTimestamp = null
    lastKeyTimestampByField = {}
    lastScrollTimestamp = null

    // Restart mouse sampling if it was stopped
    if (!mouseIntervalId) {
      mouseIntervalId = setInterval(sampleMousePosition, MOUSE_SAMPLE_INTERVAL_MS)
    }
  }

  function endCurrentSegment() {
    // Flush whatever we have in the current segment
    flushBuffers(true)

    // Stop mouse sampling while idle
    if (mouseIntervalId) {
      clearInterval(mouseIntervalId)
      mouseIntervalId = null
    }
  }

  // ── Mouse tracking ────────────────────────────────────────────────
  function handleMouseMove(event) {
    lastMouseEvent = {
      x: event.clientX,
      y: event.clientY,
      pageX: event.pageX,
      pageY: event.pageY
    }
    touchInteraction()
  }

  function sampleMousePosition() {
    if (!lastMouseEvent || isIdle || !recording) return

    mouseBuffer.push({
      x: lastMouseEvent.x,
      y: lastMouseEvent.y,
      pageX: lastMouseEvent.pageX,
      pageY: lastMouseEvent.pageY,
      t: performance.now()
    })
  }

  // ── Click tracking ────────────────────────────────────────────────
  function handleClick(event) {
    if (!recording) return
    touchInteraction()
    if (isIdle) return

    const now = performance.now()
    const dtSinceLast = lastClickTimestamp != null ? now - lastClickTimestamp : null
    lastClickTimestamp = now

    const buttonMap = { 0: 'left', 1: 'middle', 2: 'right' }

    const targetInfo = event.target ? {
      tag: event.target.tagName,
      id: event.target.id || null,
      classes: event.target.className || null,
      name: event.target.name || null,
      type: event.target.type || null,
      text: (event.target.innerText || '').slice(0, 64)
    } : null

    clickBuffer.push({
      t: now,
      x: event.clientX,
      y: event.clientY,
      button: buttonMap[event.button] || 'unknown',
      target: targetInfo,
      dt_since_last: dtSinceLast
    })
  }

  // ── Keystroke tracking (timing + non-sensitive special keys) ──────
  // Captures ALL keystrokes on the page (not just form fields) because
  // behavioral patterns like typing rhythm exist everywhere — search bars,
  // contenteditable divs, shortcut keys, etc. Only timing + special key
  // names are logged; actual characters are never recorded.

  function _getFieldId(target) {
    if (!target) return 'body'
    const tag = target.tagName || ''
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') {
      return target.name || target.id || target.type || tag.toLowerCase()
    }
    if (target.isContentEditable) return target.id || 'contenteditable'
    return target.id || tag.toLowerCase() || 'body'
  }

  function handleKeyDown(event) {
    if (!recording) return
    // Still count as interaction even if idle (will restart segment)
    touchInteraction()
    if (isIdle) return

    const now = performance.now()
    const fieldId = _getFieldId(event.target)
    const last = lastKeyTimestampByField[fieldId]
    const dt = last != null ? now - last : null
    lastKeyTimestampByField[fieldId] = now

    const key = LOGGABLE_KEYS.has(event.key) ? event.key : null

    keystrokeBuffer.push({
      field: fieldId,
      type: 'down',
      t: now,
      dt_since_last: dt,
      key: key
    })
  }

  function handleKeyUp(event) {
    if (!recording) return
    touchInteraction()
    if (isIdle) return

    const now = performance.now()
    const fieldId = _getFieldId(event.target)
    const key = LOGGABLE_KEYS.has(event.key) ? event.key : null

    keystrokeBuffer.push({
      field: fieldId,
      type: 'up',
      t: now,
      key: key
    })
  }

  // ── Scroll tracking ───────────────────────────────────────────────
  function handleScroll() {
    if (!recording) return
    touchInteraction()
    if (isIdle) return

    const now = performance.now()

    // Throttle
    if (lastScrollTimestamp && (now - lastScrollTimestamp) < SCROLL_THROTTLE_MS) return

    const sx = window.scrollX
    const sy = window.scrollY
    const dx = sx - lastScrollX
    const dy = sy - lastScrollY
    const dtSinceLast = lastScrollTimestamp != null ? now - lastScrollTimestamp : null

    lastScrollX = sx
    lastScrollY = sy
    lastScrollTimestamp = now

    scrollBuffer.push({
      t: now,
      scrollX: sx,
      scrollY: sy,
      dx: dx,
      dy: dy,
      dt_since_last: dtSinceLast
    })
  }

  // ── Visibility change (tab focus/blur) ────────────────────────────
  function handleVisibilityChange() {
    if (document.hidden) {
      pageVisible = false
      if (recording && !isIdle) {
        isIdle = true
        endCurrentSegment()
      }
    } else {
      pageVisible = true
      // Activity will resume via touchInteraction() on next user event
    }
  }

  // ── Idle detection check ──────────────────────────────────────────
  function checkIdle() {
    if (!recording || isIdle) return

    const elapsed = performance.now() - lastInteractionTime
    if (elapsed >= IDLE_THRESHOLD_MS) {
      isIdle = true
      endCurrentSegment()
    }
  }

  // ── Flush buffers to background ───────────────────────────────────
  function flushBuffers(isSegmentEnd = false) {
    if (!recording || !sessionId) return

    const hasData = mouseBuffer.length || clickBuffer.length ||
                    keystrokeBuffer.length || scrollBuffer.length

    if (!hasData && !isSegmentEnd) return

    const payload = {
      type: 'telemetry',
      sessionId: sessionId,
      segmentId: segmentId,
      tabId: null, // background will fill this
      url: window.location.href,
      hostname: window.location.hostname,
      timestamp: Date.now(),
      isSegmentEnd: isSegmentEnd,
      mouse: mouseBuffer.splice(0),
      clicks: clickBuffer.splice(0),
      keystrokes: keystrokeBuffer.splice(0),
      scroll: scrollBuffer.splice(0)
    }

    try {
      if (chrome.runtime?.id) {
        chrome.runtime.sendMessage(payload)
      }
    } catch {
      // Extension context invalidated (e.g., extension reloaded) — stop recording
      stopRecording()
    }
  }

  // ── Periodic flush timer ──────────────────────────────────────────
  let lastFlushTime = Date.now()
  function periodicFlush() {
    const now = Date.now()
    if (now - lastFlushTime >= FLUSH_INTERVAL_MS) {
      lastFlushTime = now
      flushBuffers(false)
    }
  }

  // ── Start / Stop recording ────────────────────────────────────────
  function startRecording(sid) {
    if (recording) return
    sessionId = sid
    recording = true
    isIdle = false
    segmentId = 0

    startNewSegment()

    // Send client hints and network meta once
    try {
      if (chrome.runtime?.id) {
        chrome.runtime.sendMessage({
          type: 'page_meta',
          sessionId: sessionId,
          url: window.location.href,
          hostname: window.location.hostname,
          clientHints: collectClientHints(),
          network: collectNetworkMeta(),
          timestamp: Date.now()
        })
      }
    } catch { /* ignore */ }

    // Attach event listeners
    // Use capture phase (true) for key/click events to catch them before
    // page scripts can stopPropagation(). Use document for keystrokes to
    // capture events on contenteditable, body, and non-form elements.
    window.addEventListener('mousemove', handleMouseMove, { passive: true, capture: true })
    window.addEventListener('click', handleClick, { passive: true, capture: true })
    document.addEventListener('keydown', handleKeyDown, { capture: true })
    document.addEventListener('keyup', handleKeyUp, { capture: true })
    window.addEventListener('scroll', handleScroll, { passive: true, capture: true })
    document.addEventListener('visibilitychange', handleVisibilityChange)

    // Start timers
    mouseIntervalId = setInterval(sampleMousePosition, MOUSE_SAMPLE_INTERVAL_MS)
    flushIntervalId = setInterval(periodicFlush, 1000)
    idleCheckIntervalId = setInterval(checkIdle, 1000)
  }

  function stopRecording() {
    if (!recording) return
    recording = false

    // Final flush
    flushBuffers(true)

    // Remove listeners (must match the capture flag used in addEventListener)
    window.removeEventListener('mousemove', handleMouseMove, { capture: true })
    window.removeEventListener('click', handleClick, { capture: true })
    document.removeEventListener('keydown', handleKeyDown, { capture: true })
    document.removeEventListener('keyup', handleKeyUp, { capture: true })
    window.removeEventListener('scroll', handleScroll, { capture: true })
    document.removeEventListener('visibilitychange', handleVisibilityChange)

    // Clear timers
    if (mouseIntervalId) { clearInterval(mouseIntervalId); mouseIntervalId = null }
    if (flushIntervalId) { clearInterval(flushIntervalId); flushIntervalId = null }
    if (idleCheckIntervalId) { clearInterval(idleCheckIntervalId); idleCheckIntervalId = null }
  }

  // ── Message handler (from background) ─────────────────────────────
  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === 'start_recording') {
      startRecording(msg.sessionId)
    } else if (msg.type === 'stop_recording') {
      stopRecording()
    } else if (msg.type === 'get_status') {
      // Reply with current state
      try {
        chrome.runtime.sendMessage({
          type: 'content_status',
          recording: recording,
          isIdle: isIdle,
          segmentId: segmentId,
          bufferSizes: {
            mouse: mouseBuffer.length,
            clicks: clickBuffer.length,
            keystrokes: keystrokeBuffer.length,
            scroll: scrollBuffer.length
          }
        })
      } catch { /* ignore */ }
    }
  })

  // ── Auto-start if background says we should be recording ──────────
  try {
    if (chrome.runtime?.id) {
      chrome.runtime.sendMessage({ type: 'content_ready' })
    }
  } catch { /* ignore — extension context may not be ready */ }

})()
