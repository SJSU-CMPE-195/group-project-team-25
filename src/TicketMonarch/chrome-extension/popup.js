'use strict'

const statusDot = document.getElementById('statusDot')
const statusText = document.getElementById('statusText')
const toggleBtn = document.getElementById('toggleBtn')
const exportBtn = document.getElementById('exportBtn')
const clearBtn = document.getElementById('clearBtn')
const sessionIdDisplay = document.getElementById('sessionIdDisplay')
const segmentCountEl = document.getElementById('segmentCount')
const mouseCountEl = document.getElementById('mouseCount')
const clickCountEl = document.getElementById('clickCount')
const keystrokeCountEl = document.getElementById('keystrokeCount')
const scrollCountEl = document.getElementById('scrollCount')

let isRecording = false

function formatNumber(n) {
  if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M'
  if (n >= 1000) return (n / 1000).toFixed(1) + 'K'
  return String(n)
}

function updateUI(status) {
  isRecording = status.recording

  if (isRecording) {
    statusDot.className = 'status-dot recording'
    statusText.textContent = 'Recording'
    toggleBtn.textContent = 'Stop Recording'
    toggleBtn.className = 'btn btn-stop'
    exportBtn.disabled = true
    clearBtn.disabled = true
  } else {
    statusDot.className = 'status-dot stopped'
    statusText.textContent = 'Stopped'
    toggleBtn.textContent = 'Start Recording'
    toggleBtn.className = 'btn btn-start'
    exportBtn.disabled = false
    clearBtn.disabled = false
  }

  if (status.sessionId) {
    sessionIdDisplay.textContent = status.sessionId.slice(0, 8) + '...'
    sessionIdDisplay.title = status.sessionId
  } else {
    sessionIdDisplay.textContent = '—'
  }

  segmentCountEl.textContent = formatNumber(status.segmentCount || 0)
  mouseCountEl.textContent = formatNumber(status.totalEvents?.mouse || 0)
  clickCountEl.textContent = formatNumber(status.totalEvents?.clicks || 0)
  keystrokeCountEl.textContent = formatNumber(status.totalEvents?.keystrokes || 0)
  scrollCountEl.textContent = formatNumber(status.totalEvents?.scroll || 0)
}

function fetchStatus() {
  chrome.runtime.sendMessage({ type: 'popup_get_status' }, (response) => {
    if (chrome.runtime.lastError) return
    if (response) updateUI(response)
  })
}

// Toggle recording
toggleBtn.addEventListener('click', () => {
  toggleBtn.disabled = true
  const msgType = isRecording ? 'popup_stop' : 'popup_start'

  chrome.runtime.sendMessage({ type: msgType }, () => {
    toggleBtn.disabled = false
    // Re-fetch status after a short delay to let content scripts respond
    setTimeout(fetchStatus, 300)
  })
})

// Export
exportBtn.addEventListener('click', () => {
  exportBtn.disabled = true
  exportBtn.textContent = 'Exporting...'

  chrome.runtime.sendMessage({ type: 'popup_export' }, (result) => {
    exportBtn.disabled = false
    exportBtn.textContent = 'Export JSON'

    if (result && !result.success) {
      exportBtn.textContent = result.error || 'Export failed'
      setTimeout(() => { exportBtn.textContent = 'Export JSON' }, 2000)
    }
  })
})

// Clear
clearBtn.addEventListener('click', () => {
  if (!confirm('Clear all stored telemetry data? This cannot be undone.')) return

  clearBtn.disabled = true
  chrome.runtime.sendMessage({ type: 'popup_clear' }, () => {
    clearBtn.disabled = false
    fetchStatus()
  })
})

// Initial fetch + periodic refresh while popup is open
fetchStatus()
setInterval(fetchStatus, 2000)
