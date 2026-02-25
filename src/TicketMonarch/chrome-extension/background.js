// TicketMonarch Telemetry Collector — Background Service Worker
// Manages sessions, stores telemetry segments, handles idle detection
// at the system level, and provides export functionality.
//
// MV3 constraints addressed:
//   - No Blob / URL.createObjectURL (use data-URL for export)
//   - Service worker can be killed at any time (persist all state)
//   - Storage writes must be serialized (write queue)

'use strict'

// ── In-memory cache (restored from storage on wake) ─────────────────
let sessionId = null
let recording = false
let totalEvents = { mouse: 0, clicks: 0, keystrokes: 0, scroll: 0 }
let segmentCount = 0
let stateRestored = false

// ── Write queue to prevent concurrent storage races ─────────────────
let writeQueue = Promise.resolve()

function enqueueWrite(fn) {
  writeQueue = writeQueue.then(fn).catch(err => {
    console.error('[TM background] storage write error:', err)
  })
  return writeQueue
}

// ── UUID v4 ─────────────────────────────────────────────────────────
function uuidv4() {
  return crypto.randomUUID()
}

// ── State persistence helpers ───────────────────────────────────────
async function persistCounters() {
  await chrome.storage.session.set({ totalEvents, segmentCount })
}

async function restoreState() {
  if (stateRestored) return
  try {
    const result = await chrome.storage.session.get([
      'sessionId', 'recording', 'totalEvents', 'segmentCount'
    ])
    if (result.sessionId) sessionId = result.sessionId
    if (result.recording) recording = result.recording
    if (result.totalEvents) totalEvents = result.totalEvents
    if (typeof result.segmentCount === 'number') segmentCount = result.segmentCount
  } catch { /* first run, no stored state */ }
  stateRestored = true
}

// Restore immediately on load
restoreState()

// ── Session management ──────────────────────────────────────────────
async function startNewSession() {
  sessionId = uuidv4()
  totalEvents = { mouse: 0, clicks: 0, keystrokes: 0, scroll: 0 }
  segmentCount = 0

  await chrome.storage.session.set({
    sessionId,
    recording: true,
    totalEvents,
    segmentCount,
  })

  // Initialize session entry in storage
  await enqueueWrite(async () => {
    const sessionsResult = await chrome.storage.local.get('sessions')
    const sessions = sessionsResult.sessions || {}
    sessions[sessionId] = {
      startTime: Date.now(),
      segments: [],
      pageMeta: [],
    }
    await chrome.storage.local.set({ sessions })
  })

  return sessionId
}

// ── Recording control ───────────────────────────────────────────────
async function startRecording() {
  if (recording) return
  await restoreState()

  const sid = await startNewSession()
  recording = true

  // Notify all content scripts
  const tabs = await chrome.tabs.query({})
  for (const tab of tabs) {
    if (tab.id && tab.url && !tab.url.startsWith('chrome://') &&
        !tab.url.startsWith('chrome-extension://') && !tab.url.startsWith('about:')) {
      try {
        await chrome.tabs.sendMessage(tab.id, { type: 'start_recording', sessionId: sid })
      } catch { /* tab may not have content script injected yet */ }
    }
  }
}

async function stopRecording() {
  if (!recording) return
  recording = false

  await chrome.storage.session.set({ recording: false })

  const tabs = await chrome.tabs.query({})
  for (const tab of tabs) {
    if (tab.id && tab.url && !tab.url.startsWith('chrome://') &&
        !tab.url.startsWith('chrome-extension://') && !tab.url.startsWith('about:')) {
      try {
        await chrome.tabs.sendMessage(tab.id, { type: 'stop_recording' })
      } catch { /* ignore */ }
    }
  }
}

// ── Telemetry storage (serialized via write queue) ──────────────────
function storeTelemetry(data) {
  if (!recording || !sessionId) return

  // Count events immediately in memory
  totalEvents.mouse += (data.mouse || []).length
  totalEvents.clicks += (data.clicks || []).length
  totalEvents.keystrokes += (data.keystrokes || []).length
  totalEvents.scroll += (data.scroll || []).length

  const segment = {
    segmentId: data.segmentId,
    tabId: data.tabId,
    url: data.url,
    hostname: data.hostname,
    timestamp: data.timestamp,
    isSegmentEnd: data.isSegmentEnd || false,
    mouse: data.mouse || [],
    clicks: data.clicks || [],
    keystrokes: data.keystrokes || [],
    scroll: data.scroll || [],
  }

  const hasData = segment.mouse.length || segment.clicks.length ||
                  segment.keystrokes.length || segment.scroll.length
  if (!hasData) return

  // Serialize the storage write to prevent races
  enqueueWrite(async () => {
    const sessionsResult = await chrome.storage.local.get('sessions')
    const sessions = sessionsResult.sessions || {}
    const session = sessions[sessionId]
    if (!session) return

    session.segments.push(segment)
    segmentCount = session.segments.length

    sessions[sessionId] = session
    await chrome.storage.local.set({ sessions })
    await persistCounters()
  })
}

function storePageMeta(data) {
  if (!recording || !sessionId) return

  enqueueWrite(async () => {
    const sessionsResult = await chrome.storage.local.get('sessions')
    const sessions = sessionsResult.sessions || {}
    const session = sessions[sessionId]
    if (!session) return

    session.pageMeta.push({
      url: data.url,
      hostname: data.hostname,
      clientHints: data.clientHints,
      network: data.network,
      timestamp: data.timestamp,
    })

    sessions[sessionId] = session
    await chrome.storage.local.set({ sessions })
  })
}

// ── Export (data-URL approach — Blob/createObjectURL unavailable) ────
async function exportData() {
  const sessionsResult = await chrome.storage.local.get('sessions')
  const sessions = sessionsResult.sessions || {}

  if (Object.keys(sessions).length === 0) {
    return { success: false, error: 'No data to export' }
  }

  // Consolidate segments per session: merge consecutive flushes
  // within the same segment_id into single segments
  const consolidated = {}
  for (const [sid, session] of Object.entries(sessions)) {
    const mergedSegments = []
    let currentSeg = null

    for (const seg of session.segments) {
      if (currentSeg && currentSeg.segmentId === seg.segmentId) {
        currentSeg.mouse.push(...seg.mouse)
        currentSeg.clicks.push(...seg.clicks)
        currentSeg.keystrokes.push(...seg.keystrokes)
        currentSeg.scroll.push(...seg.scroll)
        currentSeg.endTime = seg.timestamp
      } else {
        if (currentSeg) mergedSegments.push(currentSeg)
        currentSeg = {
          segmentId: seg.segmentId,
          tabId: seg.tabId,
          url: seg.url,
          hostname: seg.hostname,
          startTime: seg.timestamp,
          endTime: seg.timestamp,
          mouse: [...seg.mouse],
          clicks: [...seg.clicks],
          keystrokes: [...seg.keystrokes],
          scroll: [...seg.scroll],
        }
      }
    }
    if (currentSeg) mergedSegments.push(currentSeg)

    consolidated[sid] = {
      sessionId: sid,
      startTime: session.startTime,
      pageMeta: session.pageMeta,
      totalSegments: mergedSegments.length,
      segments: mergedSegments,
    }
  }

  // Build a data-URL (service workers can't use Blob/createObjectURL)
  const jsonStr = JSON.stringify(consolidated, null, 2)
  const bytes = new TextEncoder().encode(jsonStr)
  let binary = ''
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i])
  }
  const base64 = btoa(binary)
  const dataUrl = `data:application/json;base64,${base64}`

  const filename = `telemetry_export_${new Date().toISOString().replace(/[:.]/g, '-')}.json`

  try {
    const downloadId = await chrome.downloads.download({
      url: dataUrl,
      filename,
      saveAs: true,
    })
    if (downloadId === undefined) {
      return { success: false, error: chrome.runtime.lastError?.message || 'Download failed' }
    }
    return { success: true, filename }
  } catch (err) {
    return { success: false, error: err.message || 'Download failed' }
  }
}

async function clearData() {
  await chrome.storage.local.remove('sessions')
  totalEvents = { mouse: 0, clicks: 0, keystrokes: 0, scroll: 0 }
  segmentCount = 0
  await persistCounters()
}

// ── Message handling ────────────────────────────────────────────────
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  // Ensure state is restored before handling messages
  const handle = async () => {
    await restoreState()
    const tabId = sender.tab ? sender.tab.id : null

    switch (msg.type) {
      case 'telemetry':
        msg.tabId = tabId
        storeTelemetry(msg)
        return { success: true }

      case 'page_meta':
        storePageMeta(msg)
        return { success: true }

      case 'content_ready':
        if (recording && sessionId && tabId) {
          try {
            await chrome.tabs.sendMessage(tabId, { type: 'start_recording', sessionId })
          } catch { /* ignore */ }
        }
        return { success: true }

      case 'popup_get_status':
        return {
          recording,
          sessionId,
          totalEvents,
          segmentCount,
        }

      case 'popup_start':
        await startRecording()
        return { success: true }

      case 'popup_stop':
        await stopRecording()
        return { success: true }

      case 'popup_export':
        return await exportData()

      case 'popup_clear':
        await clearData()
        return { success: true }

      default:
        return null
    }
  }

  handle().then(result => {
    if (result !== null && result !== undefined) {
      sendResponse(result)
    }
  }).catch(err => {
    console.error('[TM background] message handler error:', err)
    sendResponse({ success: false, error: err.message })
  })

  return true // keep message channel open for async sendResponse
})

// ── System idle detection (secondary signal) ────────────────────────
try {
  chrome.idle.setDetectionInterval(15)

  chrome.idle.onStateChanged.addListener((state) => {
    if (!recording) return

    if (state === 'idle' || state === 'locked') {
      if (sessionId) {
        storePageMeta({
          url: 'system',
          hostname: 'system',
          clientHints: null,
          network: null,
          timestamp: Date.now(),
          idleState: state,
        })
      }
    }
  })
} catch { /* idle API may not be available in all contexts */ }
