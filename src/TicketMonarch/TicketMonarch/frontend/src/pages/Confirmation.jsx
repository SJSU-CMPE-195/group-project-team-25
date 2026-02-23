import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import './Confirmation.css'

function Confirmation() {
  const navigate = useNavigate()

  useEffect(() => {
    // Check if order details exist, if not redirect to home
    const details = localStorage.getItem('orderDetails')
    if (!details) {
      navigate('/')
      return
    }
  }, [navigate])

  return (
    <div className="confirmation-container">
      <header className="confirmation-header">
        <div className="confirmation-header-top">
          <div className="logo">
            <span className="logo-icon">🦋👑</span>
            <span className="logo-text">Ticket Monarch</span>
          </div>
          <div className="header-icons">
            <span className="icon">🛒</span>
            <span className="icon">☰</span>
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
        </div>
      </main>
    </div>
  )
}

export default Confirmation
