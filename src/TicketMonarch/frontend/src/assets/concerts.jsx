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

// `price` is the base ticket price for mid-level (200s) sections.
// SeatSelection derives all tier prices from this value using multipliers.
export const concerts = [
  {
    id: 1,
    name: 'Chappell Roan',
    date: 'Feb 14, 2026',
    eventName: 'Midwest Princess Tour',
    venue: 'Aragon Ballroom',
    city: 'Chicago, IL',
    location: 'Chicago, IL • Aragon Ballroom',
    image: chappellImg,
    price: 100,
  },
  {
    id: 2,
    name: 'Metallica',
    date: 'Jun 16, 2026',
    eventName: 'M72 World Tour',
    venue: 'AT&T Stadium',
    city: 'Dallas, TX',
    location: 'Dallas, TX • AT&T Stadium',
    image: metallicaImg,
    price: 250,
  },
  {
    id: 3,
    name: 'Lady Gaga',
    date: 'Oct 16, 2026',
    eventName: 'From Zero Tour',
    venue: 'Dodger Stadium',
    city: 'Los Angeles, CA',
    location: 'Los Angeles, CA • Dodger Stadium',
    image: gagaImg,
    price: 200,
  },
  {
    id: 4,
    name: 'Linkin Park',
    date: 'Nov 25, 2026',
    eventName: 'From Zero Tour',
    venue: 'AT&T Stadium',
    city: 'Dallas, TX',
    location: 'Dallas, TX • AT&T Stadium',
    image: linkinImg,
    price: 180,
  },
  {
    id: 5,
    name: 'Taylor Swift',
    date: 'Nov 13, 2026',
    eventName: 'Abono Banamex Plus Corona Capital 2025',
    venue: 'Levis Stadium',
    city: 'Santa Clara, CA',
    location: 'Santa Clara, CA • Levis Stadium',
    image: taylorImg,
    price: 300,
  },
  {
    id: 6,
    name: 'Noah Kahan',
    date: 'Jul 30, 2026',
    eventName: 'The Great Divide Tour',
    venue: 'Fenway Park',
    city: 'Boston, MA',
    location: 'Boston, MA • Fenway Park',
    image: kahanImg,
    price: 100,
  },
  {
    id: 7,
    name: 'Hozier',
    date: 'Aug 20, 2026',
    eventName: 'Unreal Unearth Tour',
    venue: 'Credit Union 1 Amphitheatre',
    city: 'Tinley Park, IL',
    location: 'Tinley Park, IL • Credit Union 1 Amphitheatre',
    image: hozierImg,
    price: 100,
  },
  {
    id: 8,
    name: 'Billy Joel',
    date: 'Dec 1, 2026',
    eventName: 'Billy Joel & Sting',
    venue: 'Rice-Eccles Stadium',
    city: 'Salt Lake City, UT',
    location: 'Salt Lake City, UT • Rice-Eccles Stadium',
    image: joelImg,
    price: 80,
  },
  {
    id: 9,
    name: 'EsDeeKid',
    date: 'Jan 1, 2027',
    eventName: 'Rolling Loud Orlando',
    venue: 'Camping World Stadium',
    city: 'Orlando, FL',
    location: 'Orlando, FL • Camping World Stadium',
    image: kidImg,
    price: 50,
  },
  {
    id: 10,
    name: 'Zara Larsson',
    date: 'Jul 30, 2026',
    eventName: 'Midnight Sun Tour',
    venue: 'House of Blues Anaheim',
    city: 'Anaheim, CA',
    location: 'Anaheim, CA • House of Blues Anaheim',
    image: larssonImg,
    price: 80,
  },
]

// Keyed by id for lookup
export const concertsById = Object.fromEntries(concerts.map(c => [c.id, c]))