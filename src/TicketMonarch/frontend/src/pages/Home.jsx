import { Link } from 'react-router-dom'
import { useState } from 'react'
import './Home.css'
import { concerts } from '../assets/concerts'

import chappellImg from '../assets/images/chappell.jpg'
import metallicaImg from '../assets/images/metallica.webp'
import gagaImg from '../assets/images/gaga.webp'
import linkinImg from '../assets/images/linkin.jpg'
import taylorImg from '../assets/images/taylor.png'
import kahanImg from '../assets/images/kahan.png'
import hozierImg from '../assets/images/hozier.png'
import joelImg from '../assets/images/joel.jpg'
import kidImg from '../assets/images/kid.jpg'
import larssonImg from '../assets/images/larsson.jpg'

function Home() {
    const [search, setSearch] = useState("")

    const filteredConcerts = concerts.filter(concert =>
    concert.name.toLowerCase().includes(search.toLowerCase())
    )

  return (
    <div className="home-container">
      <header className="home-header">
        <div className="home-header-top">
        <div className="logo">
          <span className="logo-icon">🦋</span>
          <span className="logo-text">Ticket Monarch</span>
        </div>

        <div className="search-bar">
          <span className="search-icon">🔍</span>
          <input
            type="text"
            placeholder="Search artists..."
            className="search-input"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
      </div>

        <div className="header-separator"></div>
      </header>

      <main className="home-main">
        <div className="concerts-list">

        {filteredConcerts.length === 0 && ( <p>No artists found.</p> )}
        {filteredConcerts.map(concert => (
          <div key={concert.id} className="concert-card">
            <img
              src={concert.image}
              alt={concert.name}
              className="concert-image"
            />
            <div className="concert-info">
              <h2 className="concert-name">{concert.name}</h2>
              <div className="concert-details">
                <span className="concert-date">
                  {concert.date}
                  <span className="info-icon">ℹ️</span>
                </span>
                <p className="concert-event">{concert.eventName}</p>
                <p className="concert-location">{concert.location}</p>
              </div>
            </div>

              <Link 
                to={`/seats/${concert.id}`}
                className="tickets-button"
              >
                Tickets →
              </Link>
            </div>
          ))}
        </div>
      </main>

      {/* DISCLAIMER */}
    <footer className="site-disclaimer">
      <p>
           Disclaimer: This is a student project and is not affiliated with any artists,
        venues, or event organizers. All concert information is for
        demonstration purposes only.
      </p>
      <br />
    </footer>

    </div>
  )
}

export default Home