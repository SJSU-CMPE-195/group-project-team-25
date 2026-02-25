import { v4 as uuidv4 } from 'uuid'

const API_BASE_URL = '/api'

// Target sampling rate ~50–100 Hz
const MOUSE_SAMPLE_INTERVAL_MS = 15 // ~66 Hz
const FLUSH_INTERVAL_MS = 5000 // 5 seconds

let sessionId = null
let currentPage = null
let trackingEnabled = true // false on /dev page

let mouseBuffer = []
let clickBuffer = []
let keystrokeBuffer = []
let scrollBuffer = []

let lastMouseEvent = null
let lastFlushTime = Date.now()
let lastClickTimestamp = null
let lastKeyTimestampByField = {}
let lastScrollTimestamp = null

let mouseIntervalId = null
let isInitialized = false

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

export function getSessionId() {
  if (sessionId) return sessionId

  // Try to reuse from storage if available
  const stored = window.sessionStorage.getItem('tm_session_id')
  if (stored) {
    sessionId = stored
    return sessionId
  }

  sessionId = uuidv4()
  window.sessionStorage.setItem('tm_session_id', sessionId)
  // Also store in localStorage so the /dev dashboard in another tab can find it
  window.localStorage.setItem('tm_active_session_id', sessionId)
  return sessionId
}

export function setTrackingPage(pageName) {
  currentPage = pageName
  // null means tracking is disabled (e.g. on /dev dashboard)
  trackingEnabled = pageName !== null
}

function handleRawMouseMove(event) {
  if (!trackingEnabled) return
  lastMouseEvent = {
    x: event.clientX,
    y: event.clientY
  }
}

function sampleMousePosition() {
  if (!lastMouseEvent || !trackingEnabled) return

  const timestamp = performance.now()

  mouseBuffer.push({
    x: lastMouseEvent.x,
    y: lastMouseEvent.y,
    t: timestamp
  })
}

function handleClick(event) {
  if (!trackingEnabled) return

  const now = performance.now()

  const timeSinceLastClick = lastClickTimestamp != null ? now - lastClickTimestamp : null
  lastClickTimestamp = now

  const buttonMap = {
    0: 'left',
    1: 'middle',
    2: 'right'
  }

  const targetInfo = event.target
    ? {
        tag: event.target.tagName,
        id: event.target.id || null,
        classes: event.target.className || null,
        name: event.target.name || null,
        type: event.target.type || null,
        text: (event.target.innerText || '').slice(0, 64)
      }
    : null

  clickBuffer.push({
    t: now,
    x: event.clientX,
    y: event.clientY,
    button: buttonMap[event.button] || 'unknown',
    target: targetInfo,
    dt_since_last: timeSinceLastClick
  })
}

function handleWheel(event) {
  if (!trackingEnabled) return

  const now = performance.now()
  const timeSinceLastScroll = lastScrollTimestamp != null ? now - lastScrollTimestamp : null
  lastScrollTimestamp = now

  scrollBuffer.push({
    t: now,
    scrollX: window.scrollX,
    scrollY: window.scrollY,
    dy: event.deltaY,
    dt_since_last: timeSinceLastScroll
  })
}

async function flushBuffers() {
  const now = Date.now()
  const elapsed = now - lastFlushTime

  if (elapsed < FLUSH_INTERVAL_MS) {
    return
  }

  lastFlushTime = now

  if ((!mouseBuffer.length && !clickBuffer.length && !keystrokeBuffer.length && !scrollBuffer.length) || !sessionId) {
    return
  }

  const payloadBase = {
    session_id: getSessionId(),
    page: currentPage
  }

  const mousePayload = mouseBuffer.length
    ? {
        ...payloadBase,
        samples: mouseBuffer
      }
    : null

  const clickPayload = clickBuffer.length
    ? {
        ...payloadBase,
        clicks: clickBuffer
      }
    : null

  const keystrokePayload = keystrokeBuffer.length
    ? {
        ...payloadBase,
        keystrokes: keystrokeBuffer
      }
    : null

  const scrollPayload = scrollBuffer.length
    ? {
        ...payloadBase,
        scrolls: scrollBuffer
      }
    : null

  mouseBuffer = []
  clickBuffer = []
  keystrokeBuffer = []
  scrollBuffer = []

  try {
    const requests = []

    if (mousePayload) {
      requests.push(
        fetch(`${API_BASE_URL}/tracking/mouse`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(mousePayload)
        })
      )
    }

    if (clickPayload) {
      requests.push(
        fetch(`${API_BASE_URL}/tracking/clicks`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(clickPayload)
        })
      )
    }

    if (keystrokePayload) {
      requests.push(
        fetch(`${API_BASE_URL}/tracking/keystrokes`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(keystrokePayload)
        })
      )
    }

    if (scrollPayload) {
      requests.push(
        fetch(`${API_BASE_URL}/tracking/scroll`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(scrollPayload)
        })
      )
    }

    if (requests.length) {
      await Promise.allSettled(requests)
    }
  } catch {
    // Swallow errors to avoid impacting UX
  }
}

export function initTracking() {
  if (isInitialized || typeof window === 'undefined') return
  // Never initialize tracking on the dev dashboard — it would create
  // a new session ID and overwrite localStorage, hiding the real session.
  if (window.location.pathname.startsWith('/dev')) return
  isInitialized = true

  getSessionId()

  window.addEventListener('mousemove', handleRawMouseMove, { passive: true })
  window.addEventListener('click', handleClick, { passive: true })
  window.addEventListener('wheel', handleWheel, { passive: true })

  // Keystroke tracking — captures form field typing on any page
  window.addEventListener(
    'keydown',
    (event) => {
      if (!trackingEnabled) return
      const target = event.target
      if (!target) return
      const tag = target.tagName
      const isFormField =
        tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT'
      if (!isFormField) return

      const now = performance.now()
      const fieldId = target.name || target.id || 'unknown'
      const last = lastKeyTimestampByField[fieldId]
      const dt = last != null ? now - last : null
      lastKeyTimestampByField[fieldId] = now

      // Log non-sensitive special key names; letters/digits/modifiers stay null
      const key = LOGGABLE_KEYS.has(event.key) ? event.key : null

      keystrokeBuffer.push({
        field: fieldId,
        type: 'down',
        t: now,
        dt_since_last: dt,
        key: key
      })
    },
    false
  )

  window.addEventListener(
    'keyup',
    (event) => {
      if (!trackingEnabled) return
      const target = event.target
      if (!target) return
      const tag = target.tagName
      const isFormField =
        tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT'
      if (!isFormField) return

      const now = performance.now()
      const fieldId = target.name || target.id || 'unknown'
      const key = LOGGABLE_KEYS.has(event.key) ? event.key : null

      keystrokeBuffer.push({
        field: fieldId,
        type: 'up',
        t: now,
        key: key
      })
    },
    false
  )

  mouseIntervalId = window.setInterval(sampleMousePosition, MOUSE_SAMPLE_INTERVAL_MS)

  // Periodic flush
  window.setInterval(flushBuffers, 1000)
}

/**
 * Force-flush all buffered telemetry immediately.
 * Call before agent evaluation to ensure DB has latest data.
 */
export async function forceFlush() {
  lastFlushTime = 0
  await flushBuffers()
}

/**
 * Reset the session — generates a new session ID and clears all buffers.
 * Call after a successful purchase so the next flow starts fresh.
 */
export function resetSession() {
  // Clear buffers
  mouseBuffer = []
  clickBuffer = []
  keystrokeBuffer = []
  scrollBuffer = []

  // Reset timing state
  lastMouseEvent = null
  lastClickTimestamp = null
  lastKeyTimestampByField = {}
  lastScrollTimestamp = null

  // Generate new session ID
  sessionId = uuidv4()
  window.sessionStorage.setItem('tm_session_id', sessionId)
  window.localStorage.setItem('tm_active_session_id', sessionId)

  console.log(`[Tracking] Session reset — new ID: ${sessionId}`)
}

