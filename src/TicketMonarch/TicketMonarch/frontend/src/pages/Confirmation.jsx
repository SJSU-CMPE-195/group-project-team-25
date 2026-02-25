import { useEffect, useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { confirmHumanSession } from '../services/api'
import { resetSession } from '../services/tracking'
import './Confirmation.css'

function Confirmation() {
  const navigate = useNavigate()
  const [rlResult, setRlResult] = useState(null)
  const [rlError, setRlError] = useState(null)
  const calledRef = useRef(false)

  useEffect(() => {
    const details = localStorage.getItem('orderDetails')
    if (!details) {
      navigate('/')
      return
    }

    // avoid double-firing
    if (calledRef.current) return
    calledRef.current = true

    const sessionId = window.sessionStorage.getItem('tm_session_id')
    const isBot = window.sessionStorage.getItem('tm_is_bot')
    // Bot scripts set tm_is_bot, everything else is a real human
    if (sessionId && !isBot) {
      confirmHumanSession(sessionId).then((result) => {
        if (result?.success) {
          setRlResult(result)
        } else {
          setRlError(result?.error || 'Failed to confirm session')
        }
        // Reset session so next purchase starts fresh telemetry
        resetSession()
      }).catch((err) => {
        setRlError(err.message || 'Network error')
        resetSession()
      })
    } else {
      // resent session to clear any bot flags for next time, but don't call the API since we have no session or it's already flagged as bot
      resetSession()
    }
  }, [navigate])

  return (
    <div className="confirmation-container">
      <header className="confirmation-header">
        <div className="confirmation-header-top">
          <div className="logo">
            <span className="logo-icon">🦋</span>
            <span className="logo-text">Ticket Monarch</span>
          </div>
        </div>
        <div className="header-separator"></div>
      </header>

      <main className="confirmation-main">
        <div className="confirmation-content">
          <h1 className="confirmation-title">Congratulations!</h1>
          <p className="confirmation-message">
            Your tickets have been purchased they will be in your email shortly
          </p>
          <div className="confirmation-icons">
            <span className="icon-ticket">🎫</span>
            <span className="icon-checkmark">✓</span>
          </div>
          <div className="confirmation-actions">
            <button
              className="home-button"
              onClick={() => {
                localStorage.removeItem('orderDetails')
                navigate('/')
              }}
            >
              Return to Home
            </button>
          </div>

          {/* rl agent update card */}
          <div className="rl-update-card">
            <h3 className="rl-update-title">RL Agent Update</h3>
            {!rlResult && !rlError && (
              <p className="rl-loading">Sending session to RL agent...</p>
            )}
            {rlError && (
              <p className="rl-error">Could not update agent: {rlError}</p>
            )}
            {rlResult && (
              <div className="rl-update-details">
                <div className="rl-status-badge" data-status={rlResult.updated ? 'updated' : 'skipped'}>
                  {rlResult.updated ? 'Agent Updated' : 'Update Skipped'}
                </div>
                {rlResult.reason && !rlResult.updated && (
                  <p className="rl-reason">Reason: {rlResult.reason.replace(/_/g, ' ')}</p>
                )}
                {rlResult.updated && rlResult.metrics && (
                  <div className="rl-metrics">
                    <div className="rl-metric">
                      <span className="rl-metric-value">{rlResult.steps}</span>
                      <span className="rl-metric-label">Events replayed</span>
                    </div>
                    <div className="rl-metric">
                      <span className="rl-metric-value">{rlResult.metrics.policy_loss?.toFixed(4)}</span>
                      <span className="rl-metric-label">Policy loss</span>
                    </div>
                    <div className="rl-metric">
                      <span className="rl-metric-value">{rlResult.metrics.value_loss?.toFixed(4)}</span>
                      <span className="rl-metric-label">Value loss</span>
                    </div>
                    <div className="rl-metric">
                      <span className="rl-metric-value">{rlResult.metrics.entropy?.toFixed(4)}</span>
                      <span className="rl-metric-label">Entropy</span>
                    </div>
                  </div>
                )}
                {rlResult.saved_json && (
                  <p className="rl-saved">Session saved to training data</p>
                )}
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}

export default Confirmation
