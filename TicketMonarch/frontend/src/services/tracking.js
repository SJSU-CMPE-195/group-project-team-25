import { v4 as uuidv4 } from 'uuid'

const API_BASE_URL = 'http://localhost:5000'

// Target sampling rate ~50–100 Hz
const MOUSE_SAMPLE_INTERVAL_MS = 15 // ~66 Hz
const FLUSH_INTERVAL_MS = 5000 // 5 seconds

let sessionId = null
let currentPage = null

let mouseBuffer = []
let clickBuffer = []
let keystrokeBuffer = []

let lastMouseEvent = null
let lastFlushTime = Date.now()
let lastClickTimestamp = null
let lastKeyTimestampByField = {}

let mouseIntervalId = null
let isInitialized = false

function getSessionId() {
  if (sessionId) return sessionId

  // Try to reuse from storage if available
  const stored = window.sessionStorage.getItem('tm_session_id')
  if (stored) {
    sessionId = stored
    return sessionId
  }

  sessionId = uuidv4()
  window.sessionStorage.setItem('tm_session_id', sessionId)
  return sessionId
}

export function setTrackingPage(pageName) {
  currentPage = pageName
}

function handleRawMouseMove(event) {
  lastMouseEvent = {
    x: event.clientX,
    y: event.clientY
  }
}

function sampleMousePosition() {
  if (!lastMouseEvent) return

  const timestamp = performance.now()

  mouseBuffer.push({
    x: lastMouseEvent.x,
    y: lastMouseEvent.y,
    t: timestamp
  })
}

function handleClick(event) {
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

async function flushBuffers() {
  const now = Date.now()
  const elapsed = now - lastFlushTime

  if (elapsed < FLUSH_INTERVAL_MS) {
    return
  }

  lastFlushTime = now

  if ((!mouseBuffer.length && !clickBuffer.length && !keystrokeBuffer.length) || !sessionId) {
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

  mouseBuffer = []
  clickBuffer = []
  keystrokeBuffer = []

  try {
    const requests = []

    if (mousePayload) {
      requests.push(
        fetch(`${API_BASE_URL}/api/tracking/mouse`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(mousePayload)
        })
      )
    }

    if (clickPayload) {
      requests.push(
        fetch(`${API_BASE_URL}/api/tracking/clicks`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(clickPayload)
        })
      )
    }

    if (keystrokePayload) {
      requests.push(
        fetch(`${API_BASE_URL}/api/tracking/keystrokes`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(keystrokePayload)
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
  isInitialized = true

  getSessionId()

  window.addEventListener('mousemove', handleRawMouseMove, { passive: true })
  window.addEventListener('click', handleClick, { passive: true })

  // Keystroke tracking – only for form fields on checkout page
  window.addEventListener(
    'keydown',
    (event) => {
      if (currentPage !== 'checkout') return
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

      // Privacy: we never store actual key values
      keystrokeBuffer.push({
        field: fieldId,
        type: 'down',
        t: now,
        dt_since_last: dt
      })
    },
    false
  )

  window.addEventListener(
    'keyup',
    (event) => {
      if (currentPage !== 'checkout') return
      const target = event.target
      if (!target) return
      const tag = target.tagName
      const isFormField =
        tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT'
      if (!isFormField) return

      const now = performance.now()
      const fieldId = target.name || target.id || 'unknown'

      keystrokeBuffer.push({
        field: fieldId,
        type: 'up',
        t: now
      })
    },
    false
  )

  mouseIntervalId = window.setInterval(sampleMousePosition, MOUSE_SAMPLE_INTERVAL_MS)

  // Periodic flush
  window.setInterval(flushBuffers, 1000)
}

