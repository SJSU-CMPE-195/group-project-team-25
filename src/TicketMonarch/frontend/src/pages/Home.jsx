import { Link } from 'react-router-dom'
import { useState } from 'react'
import './Home.css'
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

const concerts = [
  {
    id: 1,
    name: 'Chappell Roan',
    date: 'Until Feb 14',
    eventName: 'Midwest Princess Tour',
    location: 'Chicago, IL • Aragon Ballroom',
    image: chappellImg,
    price: 100
  },
  {
    id: 2,
    name: 'Metallica',
    date: 'Until June 16',
    eventName: 'M72 World Tour',
    location: 'Dallas, TX • AT&T Stadium',
    image: metallicaImg,
    price: 250
  },
  {
    id: 3,
    name: 'Lady Gaga',
    date: 'Until Oct 16',
    eventName: 'From Zero Tour',
    location: 'Los Angeles, CA • Dodger Stadium',
    image: gagaImg,
    price: 200
  },
  {
    id: 4,
    name: 'Linkin Park',
    date: 'Until Nov 25',
    eventName: 'From Zero Tour',
    location: 'Dallas, TX • AT&T Stadium',
    image: linkinImg,
    price: 180
  },
  {
    id: 5,
    name: 'Taylor Swift',
    date: 'Until Nov 13',
    eventName: 'Abono Banamex Plus Corona Capital 2025',
    location: 'Santa Clara, CA • Levis Stadium',
    image: taylorImg,
    price: 300
  },
  {
    id: 6,
    name: 'Noah Kahan',
    date: 'Until July 30',
    eventName: 'The Great Divide Tour',
    location: 'Boston, MA • Fenway Park',
    image: kahanImg,
    price: 100
  },
  {
    id: 7,
    name: 'Hozier',
    date: 'Until Aug 20',
    eventName: 'Unreal Unearth Tour',
    location: 'Tinley Park, IL • Credit Union 1 Amphitheatre',
    image: hozierImg,
    price: 100
  },
  {
     id: 8,
     name: 'Billy Joel',
     date: 'Until Dec 1',
     eventName: 'Billy Joel & Sting',
     location: 'Salt Lake City, UT • Rice-Eccles Stadium',
     image: joelImg,
     price: 80
  },
  {
    id: 9,
    name: 'EsDeeKid',
    date: 'Until Jan 1',
    eventName: 'Rolling Loud Orlando',
    location: 'Orlando, FL • Camping World Stadium',
    image: kidImg,
    price: 50
  },
  {
    id: 10,
    name: 'Zara Larsson',
    date: 'Until July 30',
    eventName: 'Midnight Sun Tour',
    location: 'Anaheim, CA • House of Blues Anaheim',
    image: larssonImg,
    price: 80
    }
]

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