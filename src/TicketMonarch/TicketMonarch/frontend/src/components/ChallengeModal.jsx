import { useState, useRef, useEffect, useCallback } from 'react'
import './ChallengeModal.css'

// Easy: Slider Challenge 
// Drag a slider thumb to a randomly placed target zone.

function SliderChallenge({ onComplete }) {
  const trackRef = useRef(null)
  const [thumbPos, setThumbPos] = useState(0) // 0-100%
  const [dragging, setDragging] = useState(false)
  const [attempts, setAttempts] = useState(0)
  const [feedback, setFeedback] = useState('')
  const [targetCenter] = useState(() => 30 + Math.random() * 50) // 30-80%
  const targetWidth = 8 // plus/minus 4% tolerance
  const maxAttempts = 3

  const getPercentFromEvent = useCallback((e) => {
    if (!trackRef.current) return 0
    const rect = trackRef.current.getBoundingClientRect()
    const clientX = e.touches ? e.touches[0].clientX : e.clientX
    const pct = ((clientX - rect.left) / rect.width) * 100
    return Math.max(0, Math.min(100, pct))
  }, [])

  const handleStart = useCallback((e) => {
    e.preventDefault()
    setDragging(true)
    setFeedback('')
    setThumbPos(getPercentFromEvent(e))
  }, [getPercentFromEvent])

  const handleMove = useCallback((e) => {
    if (!dragging) return
    e.preventDefault()
    setThumbPos(getPercentFromEvent(e))
  }, [dragging, getPercentFromEvent])

  const handleEnd = useCallback(() => {
    if (!dragging) return
    setDragging(false)

    const diff = Math.abs(thumbPos - targetCenter)
    if (diff <= targetWidth / 2) {
      onComplete(true)
    } else {
      const newAttempts = attempts + 1
      setAttempts(newAttempts)
      if (newAttempts >= maxAttempts) {
        onComplete(false)
      } else {
        setFeedback(`Not quite! ${maxAttempts - newAttempts} attempt(s) left.`)
        setThumbPos(0)
      }
    }
  }, [dragging, thumbPos, targetCenter, targetWidth, attempts, maxAttempts, onComplete])

  useEffect(() => {
    if (dragging) {
      window.addEventListener('mousemove', handleMove)
      window.addEventListener('mouseup', handleEnd)
      window.addEventListener('touchmove', handleMove, { passive: false })
      window.addEventListener('touchend', handleEnd)
      return () => {
        window.removeEventListener('mousemove', handleMove)
        window.removeEventListener('mouseup', handleEnd)
        window.removeEventListener('touchmove', handleMove)
        window.removeEventListener('touchend', handleEnd)
      }
    }
  }, [dragging, handleMove, handleEnd])

  return (
    <div className="challenge-inner">
      <p className="challenge-instruction">Drag the slider to the highlighted zone.</p>
      <div
        className="slider-track"
        ref={trackRef}
        onMouseDown={handleStart}
        onTouchStart={handleStart}
      >
        <div
          className="slider-target"
          style={{
            left: `${targetCenter - targetWidth / 2}%`,
            width: `${targetWidth}%`,
          }}
        />
        <div
          className="slider-thumb"
          style={{ left: `${thumbPos}%` }}
        />
      </div>
      {feedback && <p className="attempt-warning">{feedback}</p>}
    </div>
  )
}


// Medium: Canvas Distorted Text 
// distorted text on a canvas user must type what they see.


const SAFE_CHARS = 'ABCDEFGHJKMNPQRSTUVWXYZ23456789' // no 0/O, 1/l/I

function generateRandomText(length = 5) {
  let result = ''
  for (let i = 0; i < length; i++) {
    result += SAFE_CHARS[Math.floor(Math.random() * SAFE_CHARS.length)]
  }
  return result
}

function drawDistortedText(canvas, text) {
  const ctx = canvas.getContext('2d')
  const W = canvas.width
  const H = canvas.height

  // Background
  ctx.fillStyle = '#f0f0f0'
  ctx.fillRect(0, 0, W, H)

  // Background noise dots
  for (let i = 0; i < 150; i++) {
    ctx.fillStyle = `rgba(${Math.random() * 200}, ${Math.random() * 200}, ${Math.random() * 200}, 0.4)`
    ctx.fillRect(Math.random() * W, Math.random() * H, 2, 2)
  }

  // Draw each character with random transforms
  const charWidth = W / (text.length + 1)
  const fonts = ['Arial', 'Georgia', 'Courier New', 'Verdana', 'Times New Roman']
  for (let i = 0; i < text.length; i++) {
    ctx.save()
    const x = charWidth * (i + 0.5)
    const y = H / 2 + (Math.random() - 0.5) * 20
    const angle = (Math.random() - 0.5) * 0.5 // ±~25 degrees
    const fontSize = 28 + Math.random() * 12

    ctx.translate(x, y)
    ctx.rotate(angle)

    const r = Math.floor(Math.random() * 150)
    const g = Math.floor(Math.random() * 100)
    const b = Math.floor(Math.random() * 150)
    ctx.fillStyle = `rgb(${r}, ${g}, ${b})`
    ctx.font = `bold ${fontSize}px ${fonts[Math.floor(Math.random() * fonts.length)]}`
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(text[i], 0, 0)

    ctx.restore()
  }

  // Noise lines
  for (let i = 0; i < 6; i++) {
    ctx.beginPath()
    ctx.moveTo(Math.random() * W, Math.random() * H)

    // Curved noise lines
    const cp1x = Math.random() * W
    const cp1y = Math.random() * H
    const cp2x = Math.random() * W
    const cp2y = Math.random() * H
    ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, Math.random() * W, Math.random() * H)

    ctx.strokeStyle = `rgba(${Math.random() * 200}, ${Math.random() * 200}, ${Math.random() * 200}, 0.7)`
    ctx.lineWidth = 1 + Math.random() * 2
    ctx.stroke()
  }
}

function CanvasTextChallenge({ onComplete }) {
  const canvasRef = useRef(null)
  const [text, setText] = useState(() => generateRandomText(5))
  const [userInput, setUserInput] = useState('')
  const [attempts, setAttempts] = useState(0)
  const [feedback, setFeedback] = useState('')
  const maxAttempts = 3

  useEffect(() => {
    if (canvasRef.current) {
      drawDistortedText(canvasRef.current, text)
    }
  }, [text])

  const handleSubmit = (e) => {
    e.preventDefault()
    if (userInput.trim().toUpperCase() === text) {
      onComplete(true)
    } else {
      const newAttempts = attempts + 1
      setAttempts(newAttempts)
      if (newAttempts >= maxAttempts) {
        onComplete(false)
      } else {
        setFeedback(`Incorrect. ${maxAttempts - newAttempts} attempt(s) left.`)
        setUserInput('')
        // Generate new text on failure
        const newText = generateRandomText(5)
        setText(newText)
      }
    }
  }

  return (
    <div className="challenge-inner">
      <p className="challenge-instruction">Type the characters shown in the image below.</p>
      <canvas
        ref={canvasRef}
        width={300}
        height={100}
        className="captcha-canvas"
      />
      <form onSubmit={handleSubmit} className="captcha-form">
        <input
          type="text"
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          placeholder="Type characters here"
          autoFocus
          maxLength={6}
          autoComplete="off"
          spellCheck="false"
        />
        <button type="submit" className="challenge-btn">Submit</button>
      </form>
      {feedback && <p className="attempt-warning">{feedback}</p>}
    </div>
  )
}


// Hard: Timed Click Targets
// Click 4 numbered circles in order within 10 seconds.

function generateCircles(count = 4, canvasW = 400, canvasH = 300) {
  const radius = 25
  const padding = 35
  const circles = []
  for (let i = 0; i < count; i++) {
    let x, y, overlapping
    let tries = 0
    do {
      x = padding + Math.random() * (canvasW - padding * 2)
      y = padding + Math.random() * (canvasH - padding * 2)
      overlapping = circles.some(
        (c) => Math.sqrt((c.x - x) ** 2 + (c.y - y) ** 2) < radius * 2.5
      )
      tries++
    } while (overlapping && tries < 50)
    circles.push({ x, y, number: i + 1, clicked: false })
  }
  return circles
}

function drawCircles(canvas, circles, timeLeft) {
  const ctx = canvas.getContext('2d')
  const W = canvas.width
  const H = canvas.height

  // Background
  ctx.fillStyle = '#fafafa'
  ctx.fillRect(0, 0, W, H)

  // Draw border
  ctx.strokeStyle = '#ddd'
  ctx.lineWidth = 2
  ctx.strokeRect(1, 1, W - 2, H - 2)

  // Draw circles
  for (const c of circles) {
    ctx.beginPath()
    ctx.arc(c.x, c.y, 25, 0, Math.PI * 2)

    if (c.clicked) {
      ctx.fillStyle = '#4CAF50'
      ctx.fill()
      ctx.strokeStyle = '#388E3C'
    } else {
      ctx.fillStyle = '#4a90d9'
      ctx.fill()
      ctx.strokeStyle = '#2c6fad'
    }
    ctx.lineWidth = 2
    ctx.stroke()

    // Number
    ctx.fillStyle = 'white'
    ctx.font = 'bold 20px Arial'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(c.number.toString(), c.x, c.y)
  }

  // Timer
  const timerColor = timeLeft <= 3 ? '#e74c3c' : '#333'
  ctx.fillStyle = timerColor
  ctx.font = 'bold 18px Arial'
  ctx.textAlign = 'right'
  ctx.textBaseline = 'top'
  ctx.fillText(`${timeLeft.toFixed(1)}s`, W - 10, 10)
}

function TimedClickChallenge({ onComplete }) {
  const canvasRef = useRef(null)
  const [circles, setCircles] = useState(() => generateCircles())
  const [nextExpected, setNextExpected] = useState(1)
  const [timeLeft, setTimeLeft] = useState(10)
  const [active, setActive] = useState(true)
  const [attempts, setAttempts] = useState(0)
  const [feedback, setFeedback] = useState('')
  const maxAttempts = 2
  const timerRef = useRef(null)
  const startTimeRef = useRef(Date.now())

  // Timer countdown
  useEffect(() => {
    if (!active) return
    startTimeRef.current = Date.now()
    timerRef.current = setInterval(() => {
      const elapsed = (Date.now() - startTimeRef.current) / 1000
      const remaining = Math.max(0, 10 - elapsed)
      setTimeLeft(remaining)
      if (remaining <= 0) {
        clearInterval(timerRef.current)
        setActive(false)
        const newAttempts = attempts + 1
        if (newAttempts >= maxAttempts) {
          onComplete(false)
        } else {
          setAttempts(newAttempts)
          setFeedback(`Time's up! ${maxAttempts - newAttempts} attempt(s) left.`)
        }
      }
    }, 50)
    return () => clearInterval(timerRef.current)
  }, [active, attempts, maxAttempts, onComplete])

  // Redraw canvas
  useEffect(() => {
    if (canvasRef.current) {
      drawCircles(canvasRef.current, circles, timeLeft)
    }
  }, [circles, timeLeft])

  const handleCanvasClick = useCallback((e) => {
    if (!active) return
    const rect = canvasRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    for (const c of circles) {
      if (c.clicked) continue
      const dist = Math.sqrt((c.x - x) ** 2 + (c.y - y) ** 2)
      if (dist <= 28) { // slightly generous radius
        if (c.number === nextExpected) {
          c.clicked = true
          const newCircles = [...circles]
          setCircles(newCircles)
          if (nextExpected === 4) {
            clearInterval(timerRef.current)
            setActive(false)
            onComplete(true)
          } else {
            setNextExpected(nextExpected + 1)
          }
        } else {
          // Wrong order then we reset all clicks
          setFeedback('Wrong order! Click 1 first, then 2, 3, 4.')
          const reset = circles.map((c) => ({ ...c, clicked: false }))
          setCircles(reset)
          setNextExpected(1)
        }
        break
      }
    }
  }, [active, circles, nextExpected, onComplete])

  const handleRetry = () => {
    setCircles(generateCircles())
    setNextExpected(1)
    setTimeLeft(10)
    setActive(true)
    setFeedback('')
  }

  return (
    <div className="challenge-inner">
      <p className="challenge-instruction">
        Click the numbered circles in order (1 → 2 → 3 → 4) before time runs out!
      </p>
      <canvas
        ref={canvasRef}
        width={400}
        height={300}
        className="click-canvas"
        onClick={handleCanvasClick}
      />
      {feedback && <p className="attempt-warning">{feedback}</p>}
      {!active && attempts < maxAttempts && (
        <button className="challenge-btn" onClick={handleRetry}>Try Again</button>
      )}
    </div>
  )
}


// Main Modal 

function ChallengeModal({ type, difficulty, onComplete }) {
  if (type === 'blocked') {
    return (
      <div className="challenge-overlay">
        <div className="challenge-modal">
          <h2>Access Denied</h2>
          <p>Our system has detected unusual activity. This checkout has been blocked.</p>
          <button className="challenge-btn" onClick={() => onComplete(false)}>
            Go Back
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="challenge-overlay">
      <div className="challenge-modal">
        <h2>Verification Required</h2>
        {difficulty === 'easy' && <SliderChallenge onComplete={onComplete} />}
        {difficulty === 'medium' && <CanvasTextChallenge onComplete={onComplete} />}
        {difficulty === 'hard' && <TimedClickChallenge onComplete={onComplete} />}
        {!['easy', 'medium', 'hard'].includes(difficulty) && (
          <CanvasTextChallenge onComplete={onComplete} />
        )}
      </div>
    </div>
  )
}

export default ChallengeModal

