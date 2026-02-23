import { useEffect } from 'react'
import { Routes, Route, useLocation } from 'react-router-dom'
import Home from './pages/Home'
import SeatSelection from './pages/SeatSelection'
import Checkout from './pages/Checkout'
import Confirmation from './pages/Confirmation'
import DevDashboard from './pages/DevDashboard'
import './App.css'
import { initTracking, setTrackingPage } from './services/tracking'

function AppRoutes() {
  const location = useLocation()

  useEffect(() => {
    const path = location.pathname

    // Don't track on the dev dashboard — it would create its own session
    if (path.startsWith('/dev')) {
      setTrackingPage(null)
      return
    }

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
      <Route path="/dev" element={<DevDashboard />} />
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
