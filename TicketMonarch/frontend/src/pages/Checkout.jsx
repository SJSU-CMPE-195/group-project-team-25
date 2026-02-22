import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import './Checkout.css'
import { submitCheckout, evaluateSession } from '../services/api'
import { forceFlush, getSessionId } from '../services/tracking'
import ChallengeModal from '../components/ChallengeModal'

const US_STATES = [
  'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
  'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho',
  'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana',
  'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
  'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada',
  'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
  'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon',
  'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
  'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
  'West Virginia', 'Wisconsin', 'Wyoming'
]

function Checkout() {
  const navigate = useNavigate()
  const [bookingSelection, setBookingSelection] = useState(null)
  const [formData, setFormData] = useState({
    full_name: '',
    card_number: '',
    card_expiry: '',
    card_cvv: '',
    billing_address: '',
    city: '',
    state: '',
    country: 'U.S.A.',
    zip_code: ''
  })

  const [submitMessage, setSubmitMessage] = useState('')
  const [isEvaluating, setIsEvaluating] = useState(false)
  const [challengeState, setChallengeState] = useState(null) // { type: 'blocked' | 'puzzle', difficulty }

  useEffect(() => {
    // Get booking selection from localStorage
    const selection = localStorage.getItem('bookingSelection')
    if (!selection) {
      // If no selection, redirect to home
      navigate('/')
      return
    }
    setBookingSelection(JSON.parse(selection))
  }, [navigate])

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
  }

  const proceedWithCheckout = async () => {
    try {
      const result = await submitCheckout(formData)

      if (result.success) {
        const orderDetails = {
          ...bookingSelection,
          customerInfo: formData,
          orderDate: new Date().toISOString()
        }
        localStorage.setItem('orderDetails', JSON.stringify(orderDetails))
        localStorage.removeItem('bookingSelection')
        navigate('/confirmation')
      } else {
        setSubmitMessage('Error')
      }
    } catch (error) {
      setSubmitMessage('Error')
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()

    if (!bookingSelection) {
      setSubmitMessage('Error: No booking selection found')
      return
    }

    setIsEvaluating(true)
    setSubmitMessage('')

    try {
      // Flush all pending telemetry so the agent sees everything
      await forceFlush()
      // Small delay to let the backend persist the flush
      await new Promise(r => setTimeout(r, 300))

      const sessionId = getSessionId()
      const agentResult = await evaluateSession(sessionId)

      setIsEvaluating(false)

      const decision = agentResult.decision || 'allow'
      console.log(`[RL Agent] session=${sessionId} decision=${decision} confidence=${agentResult.confidence} events=${agentResult.events_processed}`)

      if (decision === 'allow') {
        await proceedWithCheckout()
      } else if (decision === 'block') {
        // Hard block — still give a hard puzzle so humans can pass
        setChallengeState({ type: 'puzzle', difficulty: 'hard' })
      } else if (decision.includes('puzzle')) {
        // easy_puzzle, medium_puzzle, hard_puzzle
        const difficulty = decision.replace('_puzzle', '') || 'medium'
        setChallengeState({ type: 'puzzle', difficulty })
      } else {
        // Unknown decision — fail open
        await proceedWithCheckout()
      }
    } catch (error) {
      setIsEvaluating(false)
      // Fail open on error
      await proceedWithCheckout()
    }
  }

  const handleChallengeComplete = async (passed) => {
    setChallengeState(null)
    if (passed) {
      await proceedWithCheckout()
    } else {
      setSubmitMessage('Verification failed. Please try again.')
    }
  }

  if (!bookingSelection) {
    return null
  }

  // Get selected section from booking selection
  const selectedSection = bookingSelection.selectedSection || 
                         (bookingSelection.seats && bookingSelection.seats[0]?.section)
  const concertName = bookingSelection.concert?.name || 'Concert'
  const ticketPrice = bookingSelection.sectionPrice || bookingSelection.total || bookingSelection.concert?.price || 0

  return (
    <div className="checkout-container">
      <header className="checkout-header">
        <div className="checkout-header-content">
          <div className="logo">
            <span className="logo-icon">🦋</span>
            <span className="logo-text">Ticket Monarch</span>
          </div>
          <div className="header-icons">
            <button 
              onClick={() => {
                // Go back to seat selection if we have concert info, otherwise go home
                if (bookingSelection?.concert?.id) {
                  navigate(`/seats/${bookingSelection.concert.id}`)
                } else {
                  navigate('/')
                }
              }}
              className="back-button-header"
            >
              ← Back
            </button>
          </div>
        </div>
      </header>
      <div className="checkout-content">
        {submitMessage && (
          <div className={`submit-message ${submitMessage === 'Submitted!' ? 'success' : 'error'}`}>
            {submitMessage}
          </div>
        )}
        
        <div className="checkout-content-wrapper">
          {/* Left Panel - Forms */}
          <div className="checkout-forms">
          <form onSubmit={handleSubmit}>
            {/* Payment Details Section */}
            <div className="form-section">
              <h2 className="section-title">Payment Details</h2>
              
              <div className="form-group">
                <label htmlFor="card_number">Card Number</label>
                <input
                  type="text"
                  id="card_number"
                  name="card_number"
                  value={formData.card_number}
                  onChange={handleChange}
                  placeholder="1234 5678 9012 3456"
                />
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label htmlFor="card_expiry">MM/YY</label>
                  <input
                    type="text"
                    id="card_expiry"
                    name="card_expiry"
                    value={formData.card_expiry}
                    onChange={handleChange}
                    placeholder="MM/YY"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="card_cvv">CVC</label>
                  <input
                    type="text"
                    id="card_cvv"
                    name="card_cvv"
                    value={formData.card_cvv}
                    onChange={handleChange}
                    placeholder="123"
                  />
                </div>
              </div>

              <div className="form-group">
                <label htmlFor="full_name">Name on Card</label>
                <input
                  type="text"
                  id="full_name"
                  name="full_name"
                  value={formData.full_name}
                  onChange={handleChange}
                  placeholder="John Doe"
                />
              </div>
            </div>

            {/* Billing Address Section */}
            <div className="form-section">
              <h2 className="section-title">Billing Address</h2>
              
              <div className="form-group">
                <label htmlFor="billing_address">Address</label>
                <input
                  type="text"
                  id="billing_address"
                  name="billing_address"
                  value={formData.billing_address}
                  onChange={handleChange}
                  placeholder="123 Main Street"
                />
              </div>

              <div className="form-group">
                <label htmlFor="apartment">Apartment, Suite, etc (optional)</label>
                <input
                  type="text"
                  id="apartment"
                  name="apartment"
                  placeholder="Apt 4B"
                />
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label htmlFor="city">City</label>
                  <input
                    type="text"
                    id="city"
                    name="city"
                    value={formData.city}
                    onChange={handleChange}
                    placeholder="New York"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="zip_code">Zip Code</label>
                  <input
                    type="text"
                    id="zip_code"
                    name="zip_code"
                    value={formData.zip_code}
                    onChange={handleChange}
                    placeholder="10001"
                  />
                </div>
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label htmlFor="country">Country</label>
                  <select
                    id="country"
                    name="country"
                    value={formData.country}
                    onChange={handleChange}
                  >
                    <option value="U.S.A.">U.S.A.</option>
                  </select>
                </div>

                <div className="form-group">
                  <label htmlFor="state">State</label>
                  <select
                    id="state"
                    name="state"
                    value={formData.state}
                    onChange={handleChange}
                  >
                    <option value="">Select State</option>
                    {US_STATES.map(state => (
                      <option key={state} value={state}>{state}</option>
                    ))}
                  </select>
                </div>
              </div>
            </div>
          </form>
          </div>

          {/* Right Panel - Purchase Details */}
          <div className="checkout-summary">
          <div className="summary-section">
            <h2 className="section-title">Purchase Details</h2>
            
            <div className="purchase-info">
              <div className="purchase-info-item">
                <span className="purchase-label">Concert:</span>
                <span className="purchase-value">{concertName}</span>
              </div>
              <div className="purchase-info-item">
                <span className="purchase-label">Section:</span>
                <span className="purchase-value">Section {selectedSection}</span>
              </div>
              <div className="purchase-info-item">
                <span className="purchase-label">Ticket Price:</span>
                <span className="purchase-value">${ticketPrice.toFixed(2)}</span>
              </div>
            </div>

            <table className="purchase-table">
              <thead>
                <tr>
                  <th>Tickets</th>
                  <th>Price</th>
                  <th>Count</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>{concertName} - Section {selectedSection}</td>
                  <td>${ticketPrice.toFixed(2)}</td>
                  <td>1</td>
                </tr>
                <tr className="total-row">
                  <td><strong>Total</strong></td>
                  <td><strong>${ticketPrice.toFixed(2)}</strong></td>
                  <td><strong>1</strong></td>
                </tr>
              </tbody>
            </table>

            <button 
              type="submit" 
              className="purchase-button"
              onClick={handleSubmit}
            >
              Purchase
            </button>
          </div>
          </div>
        </div>
      </div>

      {isEvaluating && (
        <div className="evaluating-overlay">
          <div className="evaluating-message">Verifying your session...</div>
        </div>
      )}

      {challengeState && (
        <ChallengeModal
          type={challengeState.type}
          difficulty={challengeState.difficulty}
          onComplete={handleChallengeComplete}
        />
      )}
    </div>
  )
}

export default Checkout

