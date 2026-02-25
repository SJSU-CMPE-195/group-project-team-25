import { useState, useEffect, useRef } from 'react'
import './DevDashboard.css'
import {
  fetchDashboardData,
  fetchRecentSessions,
  fetchLiveTelemetry,
  rollingEvaluate,
  confirmHumanSession,
} from '../services/api'

const ACTION_NAMES = [
  'continue', 'deploy_honeypot', 'easy_puzzle', 'medium_puzzle',
  'hard_puzzle', 'allow', 'block'
]

function DevDashboard() {
  const [sessions, setSessions] = useState([])
  // autoselect the active session from the main app tab
  const [selectedId, setSelectedId] = useState(
    () => window.localStorage.getItem('tm_active_session_id') || ''
  )
  const [liveData, setLiveData] = useState(null)
  const [rollingData, setRollingData] = useState(null)
  const [dashData, setDashData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [confirmResult, setConfirmResult] = useState(null)
  const [confirming, setConfirming] = useState(false)

  const pollRef = useRef(null)

  useEffect(() => {
    loadSessions()
    const id = setInterval(loadSessions, 5000)
    return () => clearInterval(id)
  }, [])

  const loadSessions = async () => {
    const data = await fetchRecentSessions(30)
    if (data.sessions) {
      setSessions(data.sessions)
      if (!selectedId && data.sessions.length > 0) {
        setSelectedId(data.sessions[0].session_id)
      }
    }
  }

  useEffect(() => {
    if (pollRef.current) clearInterval(pollRef.current)
    if (!selectedId || !autoRefresh) return

    const poll = async () => {
      try {
        const [live, rolling] = await Promise.all([
          fetchLiveTelemetry(selectedId),
          rollingEvaluate(selectedId),
        ])
        if (live.success) setLiveData(live)
        if (rolling.success) setRollingData(rolling)
      } catch { /* silent */ }
    }
    poll()
    pollRef.current = setInterval(poll, 2000)
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [selectedId, autoRefresh])

  useEffect(() => {
    setLiveData(null)
    setRollingData(null)
    setDashData(null)
    setConfirmResult(null)
  }, [selectedId])

  const runFullAnalysis = async () => {
    if (!selectedId) return
    setLoading(true)
    const data = await fetchDashboardData(selectedId)
    if (data) setDashData(data)
    setLoading(false)
  }

  const confirmHuman = async () => {
    if (!selectedId || confirming) return
    setConfirming(true)
    setConfirmResult(null)
    const result = await confirmHumanSession(selectedId)
    setConfirmResult(result)
    setConfirming(false)
  }

  const selectSession = (sid) => setSelectedId(sid)

  const probColor = (p) => p < 0.3 ? '#22c55e' : p < 0.6 ? '#eab308' : '#ef4444'
  const totalEvents = (s) => (s.event_counts?.mouse || 0) + (s.event_counts?.clicks || 0) + (s.event_counts?.keystrokes || 0)

  const decisionClass = (d) => {
    if (!d) return 'monitor'
    if (d === 'allow') return 'allow'
    if (d === 'block') return 'block'
    if (d.includes('puzzle')) return 'puzzle'
    return 'monitor'
  }

  return (
    <div className="dev-dashboard">
      <div className="dashboard-header">
        <h1>RL Agent Dashboard</h1>
        <div className="header-controls">
          <label className="auto-toggle">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={() => setAutoRefresh(!autoRefresh)}
            />
            Live polling
          </label>
          <button className="refresh-btn" onClick={loadSessions}>Refresh</button>
        </div>
      </div>

      <div className="dashboard-layout">
        {/* session list */}
        <div className="session-list-panel">
          <h3>Sessions ({sessions.length})</h3>
          <div className="session-list">
            {sessions.length === 0 && (
              <p className="empty-msg">No sessions in database yet.</p>
            )}
            {sessions.map(s => (
              <div
                key={s.session_id}
                className={`session-item ${selectedId === s.session_id ? 'selected' : ''}`}
                onClick={() => selectSession(s.session_id)}
              >
                <div className="session-item-id">
                  <code>{s.session_id.slice(0, 12)}...</code>
                </div>
                <div className="session-item-meta">
                  <span>{s.page || '—'}</span>
                  <span>{totalEvents(s)} events</span>
                </div>
                <div className="session-item-counts">
                  <span title="Mouse">M:{s.event_counts?.mouse || 0}</span>
                  <span title="Clicks">C:{s.event_counts?.clicks || 0}</span>
                  <span title="Keys">K:{s.event_counts?.keystrokes || 0}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* session detail */}
        <div className="session-detail-panel">
          {!selectedId ? (
            <div className="empty-detail">
              <p>Select a session from the list to view details.</p>
            </div>
          ) : (
            <>
              <div className="detail-header">
                <code className="detail-session-id">{selectedId}</code>
                <div className="detail-actions">
                  <button className="action-btn" onClick={runFullAnalysis} disabled={loading}>
                    {loading ? 'Analyzing...' : 'Run Full Analysis'}
                  </button>
                  <button className="action-btn secondary" onClick={confirmHuman} disabled={confirming}>
                    {confirming ? 'Updating...' : 'Confirm Human'}
                  </button>
                </div>
              </div>

              {/* RL Update Result, shown after Confirm Human */}
              {confirmResult && (
                <div className={`detail-section rl-confirm-result ${confirmResult.updated ? 'updated' : 'skipped'}`}>
                  <h3>{confirmResult.updated ? 'Agent Updated Successfully' : 'Update Skipped'}</h3>
                  {confirmResult.reason && !confirmResult.updated && (
                    <p className="confirm-reason">Reason: {confirmResult.reason.replace(/_/g, ' ')}</p>
                  )}
                  {confirmResult.updated && confirmResult.metrics && (
                    <div className="confirm-metrics">
                      <div className="confirm-metric">
                        <span className="confirm-metric-val">{confirmResult.steps}</span>
                        <span className="confirm-metric-lbl">Steps</span>
                      </div>
                      <div className="confirm-metric">
                        <span className="confirm-metric-val">{confirmResult.metrics.policy_loss?.toFixed(4)}</span>
                        <span className="confirm-metric-lbl">Policy Loss</span>
                      </div>
                      <div className="confirm-metric">
                        <span className="confirm-metric-val">{confirmResult.metrics.value_loss?.toFixed(4)}</span>
                        <span className="confirm-metric-lbl">Value Loss</span>
                      </div>
                      <div className="confirm-metric">
                        <span className="confirm-metric-val">{confirmResult.metrics.entropy?.toFixed(4)}</span>
                        <span className="confirm-metric-lbl">Entropy</span>
                      </div>
                      <div className="confirm-metric">
                        <span className="confirm-metric-val">{(confirmResult.online_lr * 1000).toFixed(2)}e-3</span>
                        <span className="confirm-metric-lbl">Learning Rate</span>
                      </div>
                    </div>
                  )}
                  {confirmResult.saved_json && (
                    <p className="confirm-saved">Saved: {confirmResult.saved_json}</p>
                  )}
                </div>
              )}

              {/* Live telemetry */}
              <div className="detail-section">
                <h3>Live Telemetry</h3>
                <div className="telemetry-grid">
                  <div className="telemetry-stat">
                    <span className="stat-value">{liveData?.mouse_count ?? '—'}</span>
                    <span className="stat-label">Mouse</span>
                  </div>
                  <div className="telemetry-stat">
                    <span className="stat-value">{liveData?.click_count ?? '—'}</span>
                    <span className="stat-label">Clicks</span>
                  </div>
                  <div className="telemetry-stat">
                    <span className="stat-value">{liveData?.keystroke_count ?? '—'}</span>
                    <span className="stat-label">Keys</span>
                  </div>
                  <div className="telemetry-stat">
                    <span className="stat-value">{liveData?.scroll_count ?? '—'}</span>
                    <span className="stat-label">Scrolls</span>
                  </div>
                  {(liveData?.honeypot_keystrokes > 0) && (
                    <div className="telemetry-stat honeypot-stat">
                      <span className="stat-value">{liveData.honeypot_keystrokes}</span>
                      <span className="stat-label">Honeypot</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Rolling RL inference */}
              <div className="detail-section">
                <h3>Rolling RL Inference</h3>
                {(!rollingData || rollingData.events_processed === 0) ? (
                  <p className="empty-msg">Waiting for events...</p>
                ) : (
                  <div className="rolling-card">
                    <div className="rolling-header">
                      <span>Bot Suspicion</span>
                      <span className="rolling-events">{rollingData.events_processed} events</span>
                    </div>
                    <div className="rolling-bar-bg">
                      <div
                        className="rolling-bar-fill"
                        style={{
                          width: `${Math.min((rollingData.bot_probability || 0) * 100, 100)}%`,
                          backgroundColor: probColor(rollingData.bot_probability || 0),
                        }}
                      />
                    </div>
                    <div className="rolling-prob" style={{ color: probColor(rollingData.bot_probability || 0) }}>
                      {((rollingData.bot_probability || 0) * 100).toFixed(1)}%
                    </div>
                    {rollingData.action_distribution && (
                      <div className="rolling-actions">
                        {ACTION_NAMES.map(name => {
                          const p = rollingData.action_distribution[name] || 0
                          return p > 0.01 ? (
                            <span key={name} className="rolling-action-chip">
                              {name.replace('_', ' ')}: {(p * 100).toFixed(0)}%
                            </span>
                          ) : null
                        })}
                      </div>
                    )}
                    {rollingData.deploy_honeypot && (
                      <div className="rolling-honeypot">
                        Honeypot deployed
                        {rollingData.honeypot_triggered && (
                          <span className="honeypot-caught"> — TRIGGERED</span>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Full analysis */}
              {dashData && (
                <div className="detail-section">
                  <h3>Agent Analysis</h3>

                  <div className={`decision-banner ${decisionClass(dashData.decision)}`}>
                    <span>Decision: {dashData.decision?.toUpperCase() || 'N/A'}</span>
                    <span className="decision-meta">
                      {dashData.events_processed || 0} events
                      {dashData.confidence != null && ` | ${(dashData.confidence * 100).toFixed(1)}% conf`}
                    </span>
                  </div>

                  {dashData.honeypot_triggered && (
                    <div className="honeypot-banner">
                      HONEYPOT TRIGGERED — Bot detected via hidden field
                    </div>
                  )}

                  <div className="sub-card">
                    <h4>Action Probabilities</h4>
                    <div className="action-bars">
                      {ACTION_NAMES.map((name, i) => {
                        const prob = (dashData.final_probs || [])[i] || 0
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
                  </div>

                  {(dashData.action_history || []).length > 0 && (
                    <div className="sub-card">
                      <h4>Event Timeline ({dashData.action_history.length} events)</h4>
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
                            {dashData.action_history.map((ev, i) => (
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
                    </div>
                  )}

                  {(dashData.lstm_hidden_values || []).length > 0 && (
                    <div className="sub-card">
                      <h4>LSTM Hidden State ({dashData.lstm_hidden_values.length} units)</h4>
                      <div className="hidden-state-grid">
                        {dashData.lstm_hidden_values.map((v, i) => {
                          const maxAbs = Math.max(...dashData.lstm_hidden_values.map(Math.abs), 0.01)
                          const norm = v / maxAbs
                          const r = norm >= 0 ? Math.round(40 + norm * 180) : 40
                          const b = norm < 0 ? Math.round(40 + Math.abs(norm) * 180) : 40
                          return (
                            <div
                              key={i}
                              className="hidden-cell"
                              style={{ background: `rgb(${r},40,${b})` }}
                              title={`h[${i}] = ${v.toFixed(4)}`}
                            />
                          )
                        })}
                      </div>
                      <div className="hidden-state-legend">
                        <span><span className="legend-swatch" style={{ background: '#2828dc' }} /> Negative</span>
                        <span><span className="legend-swatch" style={{ background: '#282828' }} /> Zero</span>
                        <span><span className="legend-swatch" style={{ background: '#dc2828' }} /> Positive</span>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

export default DevDashboard
