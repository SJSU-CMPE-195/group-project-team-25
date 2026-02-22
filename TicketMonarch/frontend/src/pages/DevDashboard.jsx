import { useState, useEffect, useRef, useCallback } from 'react'
import './DevDashboard.css'
import { fetchDashboardData, fetchRecentSessions, fetchLiveTelemetry } from '../services/api'

const ACTION_NAMES = [
  'continue', 'deploy_honeypot', 'easy_puzzle', 'medium_puzzle',
  'hard_puzzle', 'allow', 'block'
]

function DevDashboard() {
  const [mode, setMode] = useState('live') // 'live' or 'analyze'
  const [sessions, setSessions] = useState([])
  const [selectedSession, setSelectedSession] = useState('')
  const [dashData, setDashData] = useState(null)
  const [liveData, setLiveData] = useState(null)
  const [liveSessionId, setLiveSessionId] = useState('')
  const [loading, setLoading] = useState(false)
  const liveIntervalRef = useRef(null)
  const analyzeIntervalRef = useRef(null)

  // On mount: find the active session from the browsing tab
  useEffect(() => {
    const activeId = localStorage.getItem('tm_active_session_id')
    if (activeId) {
      setLiveSessionId(activeId)
    }
    loadSessions()
  }, [])

  // Live mode polling — fast (1s) lightweight telemetry updates
  useEffect(() => {
    if (mode === 'live' && liveSessionId) {
      const poll = async () => {
        const data = await fetchLiveTelemetry(liveSessionId)
        if (data.success) setLiveData(data)
      }
      poll() // immediate first call
      liveIntervalRef.current = setInterval(poll, 1000)
    }
    return () => {
      if (liveIntervalRef.current) clearInterval(liveIntervalRef.current)
    }
  }, [mode, liveSessionId])

  // Analyze mode auto-refresh
  useEffect(() => {
    if (mode === 'analyze' && selectedSession) {
      analyzeIntervalRef.current = setInterval(() => {
        loadDashboard(selectedSession)
      }, 3000)
    }
    return () => {
      if (analyzeIntervalRef.current) clearInterval(analyzeIntervalRef.current)
    }
  }, [mode, selectedSession])

  const loadSessions = async () => {
    const data = await fetchRecentSessions(30)
    if (data.sessions) setSessions(data.sessions)
  }

  const loadDashboard = async (sessionId) => {
    if (!sessionId) return
    setLoading(true)
    const data = await fetchDashboardData(sessionId)
    if (data) setDashData(data)
    setLoading(false)
  }

  const handleSessionChange = (e) => {
    const sid = e.target.value
    setSelectedSession(sid)
    if (sid) loadDashboard(sid)
    else setDashData(null)
  }

  const refreshLiveSession = () => {
    const activeId = localStorage.getItem('tm_active_session_id')
    if (activeId) {
      setLiveSessionId(activeId)
    }
  }

  const analyzeCurrentSession = () => {
    setMode('analyze')
    setSelectedSession(liveSessionId)
    loadDashboard(liveSessionId)
  }

  const decisionClass = (decision) => {
    if (!decision) return 'monitor'
    if (decision === 'allow') return 'allow'
    if (decision === 'block') return 'block'
    if (decision.includes('puzzle')) return 'puzzle'
    return 'monitor'
  }

  // ── Live Mode Rendering ──────────────────────────────────────

  const renderLiveMode = () => {
    if (!liveSessionId) {
      return (
        <div className="no-session">
          <p>No active session detected.</p>
          <p style={{ fontSize: '0.9rem' }}>Browse the site in another tab to start tracking.</p>
          <button className="refresh-btn" onClick={refreshLiveSession}>Check Again</button>
        </div>
      )
    }

    return (
      <>
        <div className="live-session-banner">
          <div className="live-dot" />
          <span>Tracking Session: <code>{liveSessionId}</code></span>
          <button className="refresh-btn" onClick={refreshLiveSession} style={{ marginLeft: 'auto' }}>
            Resync
          </button>
        </div>

        <div className="dashboard-grid">
          <div className="dashboard-card full-width">
            <h3>Live Telemetry</h3>
            <div className="telemetry-grid">
              <div className="telemetry-stat">
                <span className="stat-value">{liveData?.mouse_count ?? '—'}</span>
                <span className="stat-label">Mouse Moves</span>
              </div>
              <div className="telemetry-stat">
                <span className="stat-value">{liveData?.click_count ?? '—'}</span>
                <span className="stat-label">Clicks</span>
              </div>
              <div className="telemetry-stat">
                <span className="stat-value">{liveData?.keystroke_count ?? '—'}</span>
                <span className="stat-label">Keystrokes</span>
              </div>
              <div className="telemetry-stat">
                <span className="stat-value">{liveData?.scroll_count ?? '—'}</span>
                <span className="stat-label">Scrolls</span>
              </div>
            </div>
          </div>

          <div className="dashboard-card">
            <h3>Session Info</h3>
            <div className="session-info-list">
              <div className="session-info-row">
                <span className="info-label">Session ID</span>
                <span className="info-value"><code>{liveSessionId.slice(0, 12)}...</code></span>
              </div>
              <div className="session-info-row">
                <span className="info-label">Current Page</span>
                <span className="info-value">{liveData?.page || '—'}</span>
              </div>
              <div className="session-info-row">
                <span className="info-label">Started</span>
                <span className="info-value">{liveData?.session_start || '—'}</span>
              </div>
              <div className="session-info-row">
                <span className="info-label">Data in DB</span>
                <span className="info-value">{liveData?.found ? 'Yes' : 'Waiting...'}</span>
              </div>
            </div>
          </div>

          <div className="dashboard-card">
            <h3>Quick Actions</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              <button
                className="action-btn"
                onClick={analyzeCurrentSession}
                disabled={!liveData?.found}
              >
                Run Agent Analysis
              </button>
              <p style={{ fontSize: '0.8rem', color: '#888', margin: 0 }}>
                Switch to Analyze mode and run the RL agent on this session's telemetry.
              </p>
            </div>
          </div>
        </div>
      </>
    )
  }

  // ── Analyze Mode Rendering ──────────────────────────────────

  const renderActionBars = () => {
    const probs = dashData.final_probs || []
    return (
      <div className="action-bars">
        {ACTION_NAMES.map((name, i) => {
          const prob = probs[i] || 0
          return (
            <div key={name} className="action-bar-row">
              <span className="action-bar-label">{name.replace('_', ' ')}</span>
              <div className="action-bar-track">
                <div
                  className={`action-bar-fill ${name}`}
                  style={{ width: `${(prob * 100).toFixed(1)}%` }}
                />
              </div>
              <span className="action-bar-value">{(prob * 100).toFixed(1)}%</span>
            </div>
          )
        })}
      </div>
    )
  }

  const renderTimeline = () => {
    const history = dashData.action_history || []
    if (history.length === 0) return <p style={{ color: '#888' }}>No events processed.</p>

    return (
      <div className="event-timeline">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Type</th>
              <th>Action</th>
              <th>Value</th>
              <th>Top Prob</th>
            </tr>
          </thead>
          <tbody>
            {history.map((ev, i) => (
              <tr key={i}>
                <td>{i + 1}</td>
                <td>
                  <span className={`event-type-badge ${ev.event_type || ''}`}>
                    {ev.event_type || '?'}
                  </span>
                </td>
                <td>{ev.action}</td>
                <td>{ev.value?.toFixed(3) ?? '-'}</td>
                <td>{ev.probs ? (Math.max(...ev.probs) * 100).toFixed(1) + '%' : '-'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    )
  }

  const renderHiddenState = () => {
    const values = dashData.lstm_hidden_values || []
    if (values.length === 0) return <p style={{ color: '#888' }}>No hidden state data.</p>

    const maxAbs = Math.max(...values.map(Math.abs), 0.01)
    return (
      <>
        <div className="hidden-state-grid">
          {values.map((v, i) => {
            const norm = v / maxAbs
            let r, g, b
            if (norm >= 0) {
              r = Math.round(40 + norm * 180)
              g = Math.round(40)
              b = Math.round(40)
            } else {
              r = Math.round(40)
              g = Math.round(40)
              b = Math.round(40 + Math.abs(norm) * 180)
            }
            return (
              <div
                key={i}
                className="hidden-cell"
                style={{ background: `rgb(${r},${g},${b})` }}
                title={`h[${i}] = ${v.toFixed(4)}`}
              />
            )
          })}
        </div>
        <div className="hidden-state-legend">
          <div className="legend-item">
            <div className="legend-swatch" style={{ background: '#2828dc' }} />
            <span>Negative</span>
          </div>
          <div className="legend-item">
            <div className="legend-swatch" style={{ background: '#282828' }} />
            <span>Zero</span>
          </div>
          <div className="legend-item">
            <div className="legend-swatch" style={{ background: '#dc2828' }} />
            <span>Positive</span>
          </div>
        </div>
      </>
    )
  }

  const renderAnalyzeMode = () => (
    <>
      <div className="dashboard-controls">
        <select value={selectedSession} onChange={handleSessionChange}>
          <option value="">-- Select a session --</option>
          {sessions.map(s => (
            <option key={s.session_id} value={s.session_id}>
              {s.session_id.slice(0, 8)}... | {s.session_start} | {s.page || '—'}
              {s.event_counts ? ` | M:${s.event_counts.mouse} C:${s.event_counts.clicks} K:${s.event_counts.keystrokes}` : ''}
            </option>
          ))}
        </select>
        <button className="refresh-btn" onClick={() => { loadSessions(); if (selectedSession) loadDashboard(selectedSession); }}>
          Refresh
        </button>
      </div>

      {loading && !dashData && <p style={{ color: '#888' }}>Loading...</p>}

      {!selectedSession && <p className="no-session">Select a session to run agent analysis.</p>}

      {dashData && (
        <>
          <div className={`decision-banner ${decisionClass(dashData.decision)}`}>
            <span>Decision: {dashData.decision?.toUpperCase() || 'N/A'} (Action {dashData.action_index})</span>
            <span className="decision-meta">
              {dashData.events_processed || 0} events processed
              {dashData.original_action && dashData.original_action !== dashData.decision
                ? ` | original: ${dashData.original_action}`
                : ''}
              {dashData.confidence != null ? ` | confidence: ${(dashData.confidence * 100).toFixed(1)}%` : ''}
            </span>
          </div>

          <div className="dashboard-grid">
            <div className="dashboard-card">
              <h3>Telemetry Summary</h3>
              <div className="telemetry-grid">
                <div className="telemetry-stat">
                  <span className="stat-value">{dashData.telemetry_summary?.mouse_count ?? 0}</span>
                  <span className="stat-label">Mouse</span>
                </div>
                <div className="telemetry-stat">
                  <span className="stat-value">{dashData.telemetry_summary?.click_count ?? 0}</span>
                  <span className="stat-label">Clicks</span>
                </div>
                <div className="telemetry-stat">
                  <span className="stat-value">{dashData.telemetry_summary?.keystroke_count ?? 0}</span>
                  <span className="stat-label">Keys</span>
                </div>
                <div className="telemetry-stat">
                  <span className="stat-value">{dashData.telemetry_summary?.scroll_count ?? 0}</span>
                  <span className="stat-label">Scrolls</span>
                </div>
              </div>
            </div>

            <div className="dashboard-card">
              <h3>Final Action Probabilities</h3>
              {renderActionBars()}
            </div>

            <div className="dashboard-card full-width">
              <h3>Event Timeline (per-event agent decisions)</h3>
              {renderTimeline()}
            </div>

            <div className="dashboard-card full-width">
              <h3>LSTM Hidden State ({dashData.lstm_hidden_values?.length || 0} units)</h3>
              {renderHiddenState()}
            </div>
          </div>
        </>
      )}
    </>
  )

  return (
    <div className="dev-dashboard">
      <div className="dashboard-header">
        <h1>RL Agent Dev Dashboard</h1>
        <div className="mode-tabs">
          <button
            className={`mode-tab ${mode === 'live' ? 'active' : ''}`}
            onClick={() => setMode('live')}
          >
            Live Monitor
          </button>
          <button
            className={`mode-tab ${mode === 'analyze' ? 'active' : ''}`}
            onClick={() => { setMode('analyze'); loadSessions(); }}
          >
            Analyze Session
          </button>
        </div>
      </div>

      {mode === 'live' ? renderLiveMode() : renderAnalyzeMode()}
    </div>
  )
}

export default DevDashboard
