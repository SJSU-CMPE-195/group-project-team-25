import { useState, useEffect, useRef, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import './Checkout.css'
import { submitCheckout, rollingEvaluate } from '../services/api'
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

const ROLLING_POLL_MS = 3000

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
    zip_code: '',
    company_url: '',
    fax_number: '',
  })

  const [submitMessage, setSubmitMessage] = useState('')
  const [challengeState, setChallengeState] = useState(null)
  const [processing, setProcessing] = useState(false)
  const [errors, setErrors] = useState({})

  // honeypot state (driven by rolling inference)
  const [honeypotDeployed, setHoneypotDeployed] = useState(false)
  const [honeypotTriggered, setHoneypotTriggered] = useState(false)
  const latestRolling = useRef(null)
  const rollingRef = useRef(null)

  useEffect(() => {
    const selection = localStorage.getItem('bookingSelection')
    if (!selection) {
      navigate('/')
      return
    }
    setBookingSelection(JSON.parse(selection))
  }, [navigate])

  // rolling inference polling — runs every 3s to check honeypot status
  const pollRolling = useCallback(async () => {
    try {
      const sessionId = getSessionId()
      if (!sessionId) return

      await forceFlush()
      const result = await rollingEvaluate(sessionId)

      if (result.success) {
        latestRolling.current = result
        if (result.deploy_honeypot && !honeypotDeployed) {
          setHoneypotDeployed(true)
        }
        if (result.honeypot_triggered) {
          setHoneypotTriggered(true)
        }
      }
    } catch {
      // silently fail 
    }
  }, [honeypotDeployed])

  useEffect(() => {
    // start rolling polling when component mounts
    rollingRef.current = setInterval(pollRolling, ROLLING_POLL_MS)
    return () => {
      if (rollingRef.current) clearInterval(rollingRef.current)
    }
  }, [pollRolling])

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
    // clear error for this field on change
    if (errors[name]) {
      setErrors(prev => ({ ...prev, [name]: null }))
    }
  }

  const validateForm = () => {
    const errs = {}

    // --- required fields ---
    if (!formData.full_name.trim()) errs.full_name = 'Name is required'
    if (!formData.card_number.trim()) errs.card_number = 'Card number is required'
    if (!formData.card_expiry.trim()) errs.card_expiry = 'Expiry is required'
    if (!formData.card_cvv.trim()) errs.card_cvv = 'CVV is required'
    if (!formData.billing_address.trim()) errs.billing_address = 'Address is required'
    if (!formData.city.trim()) errs.city = 'City is required'
    if (!formData.state) errs.state = 'State is required'
    if (!formData.zip_code.trim()) errs.zip_code = 'Zip code is required'

    // --- name: no numbers allowed ---
    if (formData.full_name.trim() && /\d/.test(formData.full_name)) {
      errs.full_name = 'Name cannot contain numbers'
    }

    // --- card number: 13-19 digits (standard range) ---
    const digits = formData.card_number.replace(/[\s-]/g, '')
    if (formData.card_number.trim() && !/^\d{13,19}$/.test(digits)) {
      errs.card_number = 'Card number must be 13–19 digits'
    }

    // --- expiry: MM/YY format ---
    if (formData.card_expiry.trim() && !/^(0[1-9]|1[0-2])\/\d{2}$/.test(formData.card_expiry.trim())) {
      errs.card_expiry = 'Use MM/YY format'
    }

    // --- CVV: 3 or 4 digits ---
    if (formData.card_cvv.trim() && !/^\d{3,4}$/.test(formData.card_cvv.trim())) {
      errs.card_cvv = 'CVV must be 3 or 4 digits'
    }

    // --- zip: 5 digits or 5+4 format ---
    if (formData.zip_code.trim() && !/^\d{5}(-\d{4})?$/.test(formData.zip_code.trim())) {
      errs.zip_code = 'Enter a valid zip code'
    }

    // --- city: no numbers ---
    if (formData.city.trim() && /\d/.test(formData.city)) {
      errs.city = 'City cannot contain numbers'
    }

    setErrors(errs)
    return Object.keys(errs).length === 0
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
    if (e) e.preventDefault()
    if (processing) return

    // validate before doing anything
    if (!validateForm()) return

    setProcessing(true)
    setSubmitMessage('')

    if (!bookingSelection) {
      setSubmitMessage('Error: No booking selection found')
      setProcessing(false)
      return
    }

    // finish the rolling polls
    if (rollingRef.current) clearInterval(rollingRef.current)

    const sessionId = getSessionId()

    // flush all remaining telemetry then run final evaluation on all the captured data
    await forceFlush()
    const rolling = await rollingEvaluate(sessionId)
    const botProb = rolling?.bot_probability || 0
    const eventsProcessed = rolling?.events_processed || 0

    console.log(`[RL] session=${sessionId} bot_prob=${(botProb * 100).toFixed(1)}% events=${eventsProcessed} honeypot=${honeypotTriggered}`)

    // Honeypot is always trustworthy so we give hard puzzle.
    if (honeypotTriggered || rolling?.honeypot_triggered) {
      setChallengeState({ type: 'puzzle', difficulty: 'hard' })
      setProcessing(false)
      return
    }

    // High suspicion → challenge based on severity
    if (botProb > 0.45) {
      const difficulty = botProb > 0.7 ? 'hard' : botProb > 0.55 ? 'medium' : 'easy'
      setChallengeState({ type: 'puzzle', difficulty })
      setProcessing(false)
      return
    }

    // Low suspicion → allow through
    await proceedWithCheckout()
    setProcessing(false)
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

  const concertName = bookingSelection.concert?.name || 'Concert'
  const seats = bookingSelection.seats || []
  const total = bookingSelection.total || 0

  // Group seats by section for the summary table
  const sectionGroups = seats.reduce((acc, seat) => {
    const key = seat.section
    if (!acc[key]) acc[key] = { section: key, price: seat.price, count: 0 }
    acc[key].count += 1
    return acc
  }, {})

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
          {/* forms */}
          <div className="checkout-forms">
          <form onSubmit={handleSubmit}>
            {/* payment details */}
            <div className="form-section">
              <h2 className="section-title">Payment Details</h2>

              <div className="form-group">
                <label htmlFor="card_number">Card Number <span className="required">*</span></label>
                <input
                  type="text"
                  id="card_number"
                  name="card_number"
                  className={errors.card_number ? 'input-error' : ''}
                  value={formData.card_number}
                  onChange={handleChange}
                  placeholder="1234 5678 9012 3456"
                />
                {errors.card_number && <span className="field-error">{errors.card_number}</span>}
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label htmlFor="card_expiry">MM/YY <span className="required">*</span></label>
                  <input
                    type="text"
                    id="card_expiry"
                    name="card_expiry"
                    className={errors.card_expiry ? 'input-error' : ''}
                    value={formData.card_expiry}
                    onChange={handleChange}
                    placeholder="MM/YY"
                  />
                  {errors.card_expiry && <span className="field-error">{errors.card_expiry}</span>}
                </div>

                <div className="form-group">
                  <label htmlFor="card_cvv">CVC <span className="required">*</span></label>
                  <input
                    type="text"
                    id="card_cvv"
                    name="card_cvv"
                    className={errors.card_cvv ? 'input-error' : ''}
                    value={formData.card_cvv}
                    onChange={handleChange}
                    placeholder="123"
                  />
                  {errors.card_cvv && <span className="field-error">{errors.card_cvv}</span>}
                </div>
              </div>

              <div className="form-group">
                <label htmlFor="full_name">Name on Card <span className="required">*</span></label>
                <input
                  type="text"
                  id="full_name"
                  name="full_name"
                  className={errors.full_name ? 'input-error' : ''}
                  value={formData.full_name}
                  onChange={handleChange}
                  placeholder="John Doe"
                />
                {errors.full_name && <span className="field-error">{errors.full_name}</span>}
              </div>
            </div>

            {/* honeypots (hidden) */}
            <div className={honeypotDeployed ? "honeypot-field honeypot-deployed" : "honeypot-field"} aria-hidden="true" tabIndex={-1}>
              <label htmlFor="company_url">Company Website</label>
              <input
                type="text"
                id="company_url"
                name="company_url"
                value={formData.company_url}
                onChange={handleChange}
                autoComplete="nope"
                tabIndex={-1}
              />
            </div>
            <div className={honeypotDeployed ? "honeypot-field honeypot-deployed" : "honeypot-field"} aria-hidden="true" tabIndex={-1}>
              <label htmlFor="fax_number">Fax Number</label>
              <input
                type="text"
                id="fax_number"
                name="fax_number"
                value={formData.fax_number}
                onChange={handleChange}
                autoComplete="nope"
                tabIndex={-1}
              />
            </div>

            {/* biling address */}
            <div className="form-section">
              <h2 className="section-title">Billing Address</h2>

              <div className="form-group">
                <label htmlFor="billing_address">Address <span className="required">*</span></label>
                <input
                  type="text"
                  id="billing_address"
                  name="billing_address"
                  className={errors.billing_address ? 'input-error' : ''}
                  value={formData.billing_address}
                  onChange={handleChange}
                  placeholder="123 Main Street"
                />
                {errors.billing_address && <span className="field-error">{errors.billing_address}</span>}
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
                  <label htmlFor="city">City <span className="required">*</span></label>
                  <input
                    type="text"
                    id="city"
                    name="city"
                    className={errors.city ? 'input-error' : ''}
                    value={formData.city}
                    onChange={handleChange}
                    placeholder="New York"
                  />
                  {errors.city && <span className="field-error">{errors.city}</span>}
                </div>

                <div className="form-group">
                  <label htmlFor="zip_code">Zip Code <span className="required">*</span></label>
                  <input
                    type="text"
                    id="zip_code"
                    name="zip_code"
                    className={errors.zip_code ? 'input-error' : ''}
                    value={formData.zip_code}
                    onChange={handleChange}
                    placeholder="10001"
                  />
                  {errors.zip_code && <span className="field-error">{errors.zip_code}</span>}
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
                  <label htmlFor="state">State <span className="required">*</span></label>
                  <select
                    id="state"
                    name="state"
                    className={errors.state ? 'input-error' : ''}
                    value={formData.state}
                    onChange={handleChange}
                  >
                    <option value="">Select State</option>
                    {US_STATES.map(state => (
                      <option key={state} value={state}>{state}</option>
                    ))}
                  </select>
                  {errors.state && <span className="field-error">{errors.state}</span>}
                </div>
              </div>
            </div>
          </form>
          </div>

          {/* right panel for purchase details */}
          <div className="checkout-summary">
          <div className="summary-section">
            <h2 className="section-title">Purchase Details</h2>

            <div className="purchase-info">
              <div className="purchase-info-item">
                <span className="purchase-label">Concert:</span>
                <span className="purchase-value">{concertName}</span>
              </div>
              <div className="purchase-info-item">
                <span className="purchase-label">Total Tickets:</span>
                <span className="purchase-value">{seats.length}</span>
              </div>
            </div>

            <table className="purchase-table">
              <thead>
                <tr>
                  <th>Section</th>
                  <th>Price ea.</th>
                  <th>Qty</th>
                  <th>Subtotal</th>
                </tr>
              </thead>
              <tbody>
                {Object.values(sectionGroups).map(group => (
                  <tr key={group.section}>
                    <td>Section {group.section}</td>
                    <td>${group.price.toFixed(2)}</td>
                    <td>{group.count}</td>
                    <td>${(group.price * group.count).toFixed(2)}</td>
                  </tr>
                ))}
                <tr className="total-row">
                  <td colSpan={3}><strong>Total</strong></td>
                  <td><strong>${total.toFixed(2)}</strong></td>
                </tr>
              </tbody>
            </table>

            <button
              type="button"
              className="purchase-button"
              onClick={handleSubmit}
              disabled={processing}
            >
              {processing ? 'Processing...' : `Purchase — $${total.toFixed(2)}`}
            </button>
          </div>
          </div>
        </div>
      </div>

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