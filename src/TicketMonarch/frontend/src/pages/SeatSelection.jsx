import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import './SeatSelection.css'
import { concertsById } from '../assets/concerts'

// Tier config: each tier's price is concert.price * multiplier, rounded to nearest $5
const TIERS = [
  {
    id: 'floor',
    label: 'Floor',
    tag: 'General Admission',
    colorClass: 'tier-floor',
    sections: ['GA1', 'GA2', 'GA3'],
    multiplier: 1.8,
  },
  {
    id: 'hundreds',
    label: '100s',
    tag: 'Lower Bowl',
    colorClass: 'tier-100',
    sections: ['101', '102', '103', '104', '105'],
    multiplier: 1.3,
  },
  {
    id: 'twohundreds',
    label: '200s',
    tag: 'Mid Level',
    colorClass: 'tier-200',
    sections: ['201', '202', '203', '204', '205'],
    multiplier: 1.0,
  },
  {
    id: 'threehundreds',
    label: '300s',
    tag: 'Upper Bowl',
    colorClass: 'tier-300',
    sections: ['301', '302', '303', '304', '305'],
    multiplier: 0.65,
  },
]

function buildSections(basePrice) {
  return TIERS.flatMap(tier =>
    tier.sections.map((num, i) => {
      // slight within-tier variance: center sections cost slightly more
      const variance = 1 + (i - Math.floor(tier.sections.length / 2)) * 0.05
      const raw = basePrice * tier.multiplier * variance
      return {
        number: num,
        tierId: tier.id,
        colorClass: tier.colorClass,
        price: Math.round(raw / 5) * 5,
      }
    })
  )
}

const MAX_TICKETS = 8

function SeatSelection() {
  const { concertId } = useParams()
  const navigate = useNavigate()
  const [cart, setCart] = useState({}) // { sectionNumber: qty }

  const concert = concertsById[concertId]

  useEffect(() => {
    if (!concert) navigate('/')
  }, [concertId, concert, navigate])

  if (!concert) return null

  const sections = buildSections(concert.price)
  const lowestPrice = Math.min(...sections.map(s => s.price))

  const getQty = (num) => cart[num] || 0
  const setQty = (num, qty) => {
    setCart(prev => {
      const next = { ...prev }
      if (qty <= 0) delete next[num]
      else next[num] = Math.min(qty, MAX_TICKETS)
      return next
    })
  }

  const cartSections = sections.filter(s => getQty(s.number) > 0)
  const total = cartSections.reduce((sum, s) => sum + s.price * getQty(s.number), 0)
  const totalTickets = cartSections.reduce((sum, s) => sum + getQty(s.number), 0)

  const handleContinue = () => {
    if (cartSections.length === 0) return
    const seats = cartSections.flatMap(s =>
      Array.from({ length: getQty(s.number) }, (_, i) => ({
        id: `${s.number}-${i + 1}`,
        section: s.number,
        row: String(Math.floor(Math.random() * 20) + 1),
        seat: String(Math.floor(Math.random() * 30) + 1),
        price: s.price,
      }))
    )
    localStorage.setItem('bookingSelection', JSON.stringify({
      concert,
      seats,
      selectedSection: cartSections.map(s => s.number).join(', '),
      sectionPrice: cartSections[0]?.price || 0,
      total,
    }))
    navigate('/checkout')
  }

  return (
    <div className="ss-container">

      {/* ── Header ── */}
      <header className="ss-header">
        <div className="ss-header-inner">
          <div className="logo">
            <span className="logo-icon">🦋</span>
            <span className="logo-text">Ticket Monarch</span>
          </div>
          <button className="ss-back-btn" onClick={() => navigate('/')}>
            ← Back to Events
          </button>
        </div>
        <div className="header-separator" />
      </header>

      {/* ── Event info bar ── */}
      <div className="ss-event-bar">
        <div className="ss-event-bar-inner">
          <div className="ss-event-info">
            <h1 className="ss-event-name">{concert.name}</h1>
            <div className="ss-event-meta">
              <span className="ss-meta-pill">📅 {concert.date}</span>
              <span className="ss-meta-pill">📍 {concert.venue}, {concert.city}</span>
              <span className="ss-meta-pill">🎵 {concert.eventName}</span>
            </div>
          </div>
          <div className="ss-price-from">
            <span className="ss-price-label">From</span>
            <span className="ss-price-value">${lowestPrice}</span>
          </div>
        </div>
      </div>

      {/* ── Two-column body ── */}
      <div className="ss-body">
        <div className="ss-body-inner">

          {/* LEFT column: venue map */}
          <div className="ss-left">

            <div className="ss-panel">
              <div className="ss-panel-header">
                <h2 className="ss-panel-title">Select Sections</h2>
                <div className="ss-legend">
                  {TIERS.map(t => (
                    <div key={t.id} className="ss-legend-item">
                      <span className={`ss-legend-dot ${t.colorClass}`} />
                      <span>{t.label} — {t.tag}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Stadium diagram */}
              <div className="ss-stadium">
                <div className="ss-stage">
                  <span>STAGE</span>
                </div>

                {TIERS.map(tier => (
                  <div key={tier.id} className="ss-tier-row">
                    <div className={`ss-tier-side-label ${tier.colorClass}-text`}>{tier.label}</div>
                    <div className="ss-section-group">
                      {sections
                        .filter(s => s.tierId === tier.id)
                        .map(section => {
                          const qty = getQty(section.number)
                          return (
                            <div
                              key={section.number}
                              className={`ss-section-cell ${section.colorClass}${qty > 0 ? ' is-selected' : ''}`}
                            >
                              <div className="ss-cell-info">
                                <span className="ss-cell-num">{section.number}</span>
                                <span className="ss-cell-price">${section.price}</span>
                              </div>
                              <div className="ss-cell-stepper">
                                <button
                                  className="ss-step-btn"
                                  onClick={() => setQty(section.number, qty - 1)}
                                  disabled={qty === 0}
                                >−</button>
                                <span className="ss-step-qty">{qty}</span>
                                <button
                                  className="ss-step-btn"
                                  onClick={() => setQty(section.number, qty + 1)}
                                  disabled={qty >= MAX_TICKETS}
                                >+</button>
                              </div>
                            </div>
                          )
                        })}
                    </div>
                  </div>
                ))}

                <div className="ss-stadium-note">
                  General seating within each section · Max {MAX_TICKETS} tickets per section
                </div>
              </div>
            </div>
          </div>

          {/* RIGHT column: order summary */}
          <div className="ss-right">
            <div className="ss-order-panel">
              <h2 className="ss-order-title">
                Order Summary
                {totalTickets > 0 && (
                  <span className="ss-order-badge">{totalTickets}</span>
                )}
              </h2>

              {cartSections.length === 0 ? (
                <div className="ss-order-empty">
                  <div className="ss-order-empty-icon">🎟️</div>
                  <p>Add tickets from the map to get started</p>
                </div>
              ) : (
                <>
                  <div className="ss-order-items">
                    {cartSections.map(s => {
                      const tier = TIERS.find(t => t.id === s.tierId)
                      return (
                        <div key={s.number} className="ss-order-item">
                          <div className="ss-order-item-left">
                            <span className={`ss-order-dot ${s.colorClass}`} />
                            <div>
                              <div className="ss-order-section-name">Section {s.number}</div>
                              <div className="ss-order-tier-name">{tier?.tag}</div>
                            </div>
                          </div>
                          <div className="ss-order-item-right">
                            <div className="ss-order-stepper">
                              <button className="ss-order-step-btn" onClick={() => setQty(s.number, getQty(s.number) - 1)}>−</button>
                              <span className="ss-order-qty">{getQty(s.number)}</span>
                              <button className="ss-order-step-btn" onClick={() => setQty(s.number, getQty(s.number) + 1)} disabled={getQty(s.number) >= MAX_TICKETS}>+</button>
                            </div>
                            <div className="ss-order-line-price">${(s.price * getQty(s.number)).toLocaleString()}</div>
                          </div>
                        </div>
                      )
                    })}
                  </div>

                  <div className="ss-order-divider" />
                  <div className="ss-total-line">
                    <span>Total</span>
                    <span>${total.toLocaleString()}</span>
                  </div>

                  <button className="ss-checkout-btn" onClick={handleContinue}>
                    Checkout — ${total.toLocaleString()}
                  </button>
                  <p className="ss-checkout-note">You won't be charged until the next step</p>
                </>
              )}
            </div>
          </div>

        </div>
      </div>
    </div>
  )
}

export default SeatSelection
