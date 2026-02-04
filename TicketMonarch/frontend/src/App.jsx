import { useEffect } from 'react'
import { Routes, Route, useLocation } from 'react-router-dom'
import Home from './pages/Home'
import SeatSelection from './pages/SeatSelection'
import Checkout from './pages/Checkout'
import Confirmation from './pages/Confirmation'
import './App.css'
import { initTracking, setTrackingPage } from './services/tracking'

function AppRoutes() {
  const location = useLocation()

  useEffect(() => {
    const path = location.pathname
    let pageName = 'home'

    if (path.startsWith('/seats')) {
      pageName = 'seat_selection'
    } else if (path.startsWith('/checkout')) {
      pageName = 'checkout'
    } else if (path.startsWith('/confirmation')) {
      pageName = 'confirmation'
    }

    setTrackingPage(pageName)
  }, [location])

  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/seats/:concertId" element={<SeatSelection />} />
      <Route path="/checkout" element={<Checkout />} />
      <Route path="/confirmation" element={<Confirmation />} />
    </Routes>
  )
}

function App() {
  useEffect(() => {
    initTracking()
  }, [])

  return <AppRoutes />
}

export default App
