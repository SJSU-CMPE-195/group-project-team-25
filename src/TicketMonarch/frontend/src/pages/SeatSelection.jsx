import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import './SeatSelection.css'

// Sample concert data - in a real app, this would come from an API
const concerts = {
   1: {
     id: 1,
     name: 'Chappell Roan',
     date: 'Until Feb 14',
     venue: 'Aragon Ballroom',
     city: 'Chicago, IL',
     price: 100
   },
   2: {
     id: 2,
     name: 'Metallica',
     date: 'Until June 16',
     venue: 'AT&T Stadium',
     city: 'Dallas, TX',
     price: 250
   },
   3: {
     id: 3,
     name: 'Lady Gaga',
     date: 'Until Oct 16',
     venue: 'Dodger Stadium',
     city: 'Los Angeles, CA',
     price: 200
   },
   4: {
     id: 4,
     name: 'Linkin Park',
     date: 'Until Nov 25',
     venue: 'AT&T Stadium',
     city: 'Dallas, TX',
     price: 180
   },
   5: {
     id: 5,
     name: 'Taylor Swift',
     date: 'Until Nov 13',
     venue: 'Levis Stadium',
     city: 'Santa Clara, CA',
     price: 300
   },
   6: {
     id: 6,
     name: 'Noah Kahan',
     date: 'Until July 30',
     venue: 'Fenway Park',
     city: 'Boston, MA',
     price: 100
   },
   7: {
     id: 7,
     name: 'Hozier',
     date: 'Until Aug 20',
     venue: 'Credit Union 1 Amphitheatre',
     city: 'Tinley Park, IL',
     price: 100
   },
   8: {
     id: 8,
     name: 'Billy Joel',
     date: 'Until Dec 1',
     venue: 'Rice-Eccles Stadium',
     city: 'Salt Lake City, UT',
     price: 80
   },
   9: {
     id: 9,
     name: 'EsDeeKid',
     date: 'Until Jan 1',
     venue: 'Camping World Stadium',
     city: 'Orlando, FL',
     price: 50
   },
   10: {
     id: 10,
     name: 'Zara Larsson',
     date: 'Until July 30',
     venue: 'House of Blues Anaheim',
     city: 'Anaheim, CA',
     price: 80
   }
 }

// Define sections with prices
const sections = [
  { number: '1', price: 20 },
    { number: '2', price: 30 },
    { number: '3', price: 40 },
    { number: '4', price: 50 },
    { number: '5', price: 60 },

    { number: '101', price: 110 },
    { number: '102', price: 120 },
    { number: '103', price: 130 },
    { number: '104', price: 140 },
    { number: '105', price: 150 },

    { number: '201', price: 210 },
    { number: '202', price: 225 },
    { number: '203', price: 240 },
    { number: '204', price: 255 },
    { number: '205', price: 270 },

    { number: '301', price: 310 },
    { number: '302', price: 330 },
    { number: '303', price: 350 },
    { number: '304', price: 370 },
    { number: '305', price: 390 },
]

function SeatSelection() {
  const { concertId } = useParams()
  const navigate = useNavigate()
  const [selectedSection, setSelectedSection] = useState(null)
  const [selectedPrice, setSelectedPrice] = useState(null)
  const concert = concerts[concertId]

  useEffect(() => {
    if (!concert) {
      navigate('/')
      return
    }
  }, [concertId, concert, navigate])

  if (!concert) {
    return null
  }

  const handleSectionClick = (sectionNumber, sectionPrice) => {
    // Toggle selection
    if (selectedSection === sectionNumber) {
      setSelectedSection(null)
      setSelectedPrice(null)
    } else {
      setSelectedSection(sectionNumber)
      setSelectedPrice(sectionPrice)
    }
  }

  const handleContinue = () => {
    if (!selectedSection || !selectedPrice) {
      alert('Please select a section')
      return
    }

    // Store selection in localStorage for checkout page
    const selection = {
      concert,
      seats: [{
        id: `section-${selectedSection}`,
        section: selectedSection,
        row: '1',
        seat: '1'
      }],
      selectedSection: selectedSection,
      total: selectedPrice,
      sectionPrice: selectedPrice
    }
    localStorage.setItem('bookingSelection', JSON.stringify(selection))
    navigate('/checkout')
  }

  const isSectionSelected = (sectionNumber) => {
    return selectedSection === sectionNumber
  }

  return (
    <div className="seat-selection-container">
      <header className="seat-selection-header">
        <div className="seat-selection-header-top">
          <div className="logo">
            <span className="logo-icon">🦋</span>
            <span className="logo-text">Ticket Monarch</span>
          </div>
          <div className="header-icons">
            <span className="icon">☰</span>
          </div>
        </div>
        <div className="header-separator"></div>
      </header>

      <main className="seat-selection-main">
        <h1 className="seat-selection-title">Seat Selection</h1>
        
        <div className="sections-grid-container">
          <div className="sections-grid">
            {sections.map(section => (
              <button
                key={section.number}
                className={`section-button ${isSectionSelected(section.number) ? 'selected' : ''}`}
                onClick={() => handleSectionClick(section.number, section.price)}
              >
                <div className="section-number">{section.number}</div>
                <div className="section-price">${section.price}</div>
              </button>
            ))}
          </div>
        </div>

        {selectedSection && (
          <div className="selection-actions">
            <button 
              className="continue-button" 
              onClick={handleContinue}
            >
              Continue to Checkout
            </button>
            <button className="back-button" onClick={() => navigate('/')}>
              ← Back to Concerts
            </button>
          </div>
        )}
      </main>
    </div>
  )
}

export default SeatSelection
