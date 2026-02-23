import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import './SeatSelection.css'

// Sample concert data - in a real app, this would come from an API
const concerts = {
  1: {
    id: 1,
    name: 'Chappell Roan',
    date: 'November 16, 2024',
    venue: 'Autodromo Hnos. Rodriguez',
    city: 'México, CDMX, Mexico',
    price: 100
  },
  2: {
    id: 2,
    name: 'Metallica',
    date: 'November 16, 2024',
    venue: 'Autodromo Hnos. Rodriguez',
    city: 'México, CDMX, Mexico',
    price: 250
  },
  3: {
    id: 3,
    name: 'Lady Gaga',
    date: 'November 16, 2024',
    venue: 'Autodromo Hnos. Rodriguez',
    city: 'México, CDMX, Mexico',
    price: 200
  },
  4: {
    id: 4,
    name: 'Linkin Park',
    date: 'November 16, 2024',
    venue: 'Autodromo Hnos. Rodriguez',
    city: 'México, CDMX, Mexico',
    price: 180
  },
  5: {
    id: 5,
    name: 'Taylor Swift',
    date: 'November 16, 2024',
    venue: 'Autodromo Hnos. Rodriguez',
    city: 'México, CDMX, Mexico',
    price: 300
  }
}

// Define sections with prices
const sections = [
  { number: '100', price: 50 },
  { number: '101', price: 75 },
  { number: '102', price: 100 },
  { number: '200', price: 150 },
  { number: '201', price: 200 },
  { number: '202', price: 250 }
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
            <span className="icon">🛒</span>
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
