import { useState, useRef, useEffect, useCallback } from 'react'
import './ChallengeModal.css'

/* EASY: Rotation Challenge
 Rotate an asymmetric object until it is upright.
 This is the lowest-friction challenge: one visual judgment and one continuous
 adjustment. The object starts at a randomized angle and the user rotates it
 back to upright within a generous tolerance. */

function drawRotationObject(ctx, size, objectIndex) {
  const cx = size / 2
  const cy = size / 2
  const r = size * 0.34

  ctx.clearRect(0, 0, size, size)

  const objects = [
    () => {
      ctx.fillStyle = '#f4d0a7'
      ctx.fillRect(cx - r * 0.7, cy - r * 0.15, r * 1.4, r * 1.05)
      ctx.beginPath()
      ctx.moveTo(cx - r * 0.9, cy - r * 0.15)
      ctx.lineTo(cx, cy - r * 0.95)
      ctx.lineTo(cx + r * 0.9, cy - r * 0.15)
      ctx.closePath()
      ctx.fillStyle = '#d04a3b'
      ctx.fill()
      ctx.fillStyle = '#85bfe7'
      ctx.fillRect(cx - r * 0.45, cy - r * 0.02, r * 0.32, r * 0.28)
      ctx.fillStyle = '#6d4c41'
      ctx.fillRect(cx - r * 0.13, cy + r * 0.14, r * 0.26, r * 0.76)
    },
    () => {
      ctx.fillStyle = '#267ccf'
      ctx.beginPath()
      ctx.moveTo(cx, cy - r)
      ctx.lineTo(cx + r * 0.58, cy)
      ctx.lineTo(cx + r * 0.24, cy)
      ctx.lineTo(cx + r * 0.24, cy + r)
      ctx.lineTo(cx - r * 0.24, cy + r)
      ctx.lineTo(cx - r * 0.24, cy)
      ctx.lineTo(cx - r * 0.58, cy)
      ctx.closePath()
      ctx.fill()
    },
    () => {
      ctx.beginPath()
      ctx.arc(cx, cy - r * 0.22, r * 0.42, Math.PI, 0)
      ctx.strokeStyle = '#f0b533'
      ctx.lineWidth = r * 0.18
      ctx.stroke()
      ctx.fillStyle = '#d3a521'
      ctx.fillRect(cx - r * 0.56, cy - r * 0.05, r * 1.12, r * 0.98)
      ctx.beginPath()
      ctx.arc(cx, cy + r * 0.18, r * 0.15, 0, Math.PI * 2)
      ctx.fillStyle = '#2c3e50'
      ctx.fill()
      ctx.fillRect(cx - r * 0.035, cy + r * 0.18, r * 0.07, r * 0.22)
    },
    () => {
      ctx.fillStyle = '#7b4f2d'
      ctx.fillRect(cx - r * 0.1, cy + r * 0.18, r * 0.2, r * 0.72)
      ctx.beginPath()
      ctx.moveTo(cx, cy - r)
      ctx.lineTo(cx + r * 0.72, cy + r * 0.15)
      ctx.lineTo(cx - r * 0.72, cy + r * 0.15)
      ctx.closePath()
      ctx.fillStyle = '#24a85a'
      ctx.fill()
      ctx.beginPath()
      ctx.moveTo(cx, cy - r * 0.52)
      ctx.lineTo(cx + r * 0.56, cy + r * 0.42)
      ctx.lineTo(cx - r * 0.56, cy + r * 0.42)
      ctx.closePath()
      ctx.fillStyle = '#35c76c'
      ctx.fill()
    },
  ]

  objects[objectIndex % objects.length]()
}

function RotationChallenge({ onComplete }) {
  const canvasRef = useRef(null)
  const [objectIndex] = useState(() => Math.floor(Math.random() * 4))
  const [baseAngle] = useState(() => {
    const angle = 40 + Math.floor(Math.random() * 100)
    return Math.random() > 0.5 ? angle : 360 - angle
  })
  const [userAngle, setUserAngle] = useState(0)
  const [attempts, setAttempts] = useState(0)
  const [feedback, setFeedback] = useState('')
  const [solved, setSolved] = useState(false)
  const maxAttempts = 3
  const size = 164

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, size, size)
    ctx.save()
    ctx.translate(size / 2, size / 2)
    ctx.rotate(((baseAngle - userAngle) * Math.PI) / 180)
    ctx.translate(-size / 2, -size / 2)
    drawRotationObject(ctx, size, objectIndex)
    ctx.restore()
  }, [baseAngle, userAngle, objectIndex])

  const nudge = useCallback((delta) => {
    setUserAngle((prev) => (prev + delta + 360) % 360)
    setFeedback('')
  }, [])

  const handleSubmit = () => {
    const diff = Math.abs(((baseAngle - userAngle) + 360) % 360)
    const isUpright = diff <= 12 || diff >= 348

    if (isUpright) {
      setSolved(true)
      setFeedback('Verified!')
      setTimeout(() => onComplete(true), 250)
      return
    }

    const nextAttempts = attempts + 1
    setAttempts(nextAttempts)

    if (nextAttempts >= maxAttempts) {
      onComplete(false)
    } else {
      setUserAngle(0)
      setFeedback(`Not quite upright. ${maxAttempts - nextAttempts} attempt(s) left.`)
    }
  }

  return (
    <div className="challenge-inner">
      <p className="challenge-instruction">
        Rotate the object until it is upright, then press <strong>Submit</strong>.
      </p>

      <div className={`rotation-wrap ${solved ? 'solved' : ''}`}>
        <canvas
          ref={canvasRef}
          width={size}
          height={size}
          className="rotation-canvas"
        />
      </div>

      <div className="rotation-controls">
        <button
          type="button"
          className="rotate-btn"
          onClick={() => nudge(-6)}
          aria-label="Rotate counter-clockwise"
        >
          ↺
        </button>

        <input
          type="range"
          className="rotation-dial"
          min="0"
          max="359"
          value={userAngle}
          onChange={(e) => {
            setUserAngle(Number(e.target.value))
            setFeedback('')
          }}
          aria-label="Rotation angle"
        />

        <button
          type="button"
          className="rotate-btn"
          onClick={() => nudge(6)}
          aria-label="Rotate clockwise"
        >
          ↻
        </button>
      </div>

      <button type="button" className="challenge-btn" onClick={handleSubmit}>
        Submit
      </button>

      {feedback && (
        <p className={`attempt-message ${feedback === 'Verified!' ? 'success' : ''}`}>
          {feedback}
        </p>
      )}
    </div>
  )
}

/* MEDIUM: Jigsaw Slider Challenge
 Drag a puzzle piece horizontally until it aligns with the missing cut-out.
 Compared with a plain target slider, this adds a visual localization step while
 remaining familiar and fairly quick for humans to solve. */

function drawScene(ctx, w, h, sceneIndex) {
  const scenes = [
    () => {
      const sky = ctx.createLinearGradient(0, 0, 0, h * 0.62)
      sky.addColorStop(0, '#18203a')
      sky.addColorStop(1, '#ef9a6d')
      ctx.fillStyle = sky
      ctx.fillRect(0, 0, w, h * 0.62)
      ctx.fillStyle = '#39617e'
      ctx.fillRect(0, h * 0.62, w, h * 0.38)
      ctx.beginPath()
      ctx.arc(w * 0.52, h * 0.54, h * 0.12, 0, Math.PI * 2)
      ctx.fillStyle = '#f8c344'
      ctx.fill()
      for (let i = 0; i < 6; i++) {
        ctx.beginPath()
        ctx.moveTo(w * 0.12 + i * w * 0.12, h * 0.72 + i * 4)
        ctx.lineTo(w * 0.28 + i * w * 0.12, h * 0.72 + i * 4)
        ctx.strokeStyle = 'rgba(255,255,255,0.35)'
        ctx.lineWidth = 1.4
        ctx.stroke()
      }
    },
    () => {
      ctx.fillStyle = '#b9d6e9'
      ctx.fillRect(0, 0, w, h)
      ctx.beginPath()
      ctx.moveTo(0, h)
      ctx.lineTo(w * 0.2, h * 0.36)
      ctx.lineTo(w * 0.44, h * 0.56)
      ctx.lineTo(w * 0.66, h * 0.25)
      ctx.lineTo(w * 0.86, h * 0.46)
      ctx.lineTo(w, h * 0.32)
      ctx.lineTo(w, h)
      ctx.fillStyle = '#5d7d63'
      ctx.fill()
      ctx.fillStyle = '#ffffff'
      ctx.beginPath()
      ctx.moveTo(w * 0.18, h * 0.42)
      ctx.lineTo(w * 0.2, h * 0.36)
      ctx.lineTo(w * 0.23, h * 0.42)
      ctx.fill()
      ctx.beginPath()
      ctx.moveTo(w * 0.63, h * 0.33)
      ctx.lineTo(w * 0.66, h * 0.25)
      ctx.lineTo(w * 0.69, h * 0.33)
      ctx.fill()
    },
    () => {
      const sky = ctx.createLinearGradient(0, 0, 0, h * 0.7)
      sky.addColorStop(0, '#14102b')
      sky.addColorStop(1, '#c96330')
      ctx.fillStyle = sky
      ctx.fillRect(0, 0, w, h * 0.7)
      ctx.fillStyle = '#17182b'
      ctx.fillRect(0, h * 0.7, w, h * 0.3)
      const buildings = [
        [0.05, 0.49, 0.08, 0.5], [0.14, 0.35, 0.09, 0.65],
        [0.25, 0.44, 0.07, 0.56], [0.33, 0.24, 0.1, 0.76],
        [0.45, 0.4, 0.08, 0.6], [0.55, 0.31, 0.12, 0.69],
        [0.7, 0.44, 0.08, 0.56], [0.8, 0.35, 0.1, 0.65],
      ]
      buildings.forEach(([x, topY, bw, botY]) => {
        ctx.fillStyle = '#0b0d18'
        ctx.fillRect(x * w, topY * h, bw * w, (botY - topY) * h + h * 0.3)
        for (let wy = topY * h + 5; wy < botY * h; wy += 8) {
          for (let wx = x * w + 3; wx < (x + bw) * w - 3; wx += 7) {
            if (Math.random() > 0.45) {
              ctx.fillStyle = 'rgba(255, 226, 116, 0.75)'
              ctx.fillRect(wx, wy, 4, 4)
            }
          }
        }
      })
    },
  ]

  scenes[sceneIndex % scenes.length]()
}

function puzzlePiecePath(ctx, x, y, pw, ph) {
  const topNotchR = ph * 0.16
  const sideTabR = ph * 0.18

  ctx.beginPath()
  ctx.moveTo(x, y)
  ctx.lineTo(x + pw * 0.5 - topNotchR, y)
  ctx.arc(x + pw * 0.5, y, topNotchR, Math.PI, 0, true)
  ctx.lineTo(x + pw, y)
  ctx.lineTo(x + pw, y + ph * 0.5 - sideTabR)
  ctx.arc(x + pw, y + ph * 0.5, sideTabR, -Math.PI / 2, Math.PI / 2, false)
  ctx.lineTo(x + pw, y + ph)
  ctx.lineTo(x, y + ph)
  ctx.closePath()
}

function JigsawSliderChallenge({ onComplete }) {
  const bgCanvasRef = useRef(null)
  const pieceCanvasRef = useRef(null)
  const trackRef = useRef(null)

  const [sceneIndex] = useState(() => Math.floor(Math.random() * 3))
  const [gapX] = useState(() => 25 + Math.floor(Math.random() * 50))
  const [sliderPct, setSliderPct] = useState(0)
  const [dragging, setDragging] = useState(false)
  const [attempts, setAttempts] = useState(0)
  const [feedback, setFeedback] = useState('')
  const [solved, setSolved] = useState(false)
  const maxAttempts = 3

  const canvasW = 320
  const canvasH = 124
  const pieceW = 52
  const pieceH = 52
  const pieceY = (canvasH - pieceH) / 2

  useEffect(() => {
    const canvas = bgCanvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    drawScene(ctx, canvasW, canvasH, sceneIndex)

    const gapPxX = (gapX / 100) * canvasW
    ctx.save()
    puzzlePiecePath(ctx, gapPxX, pieceY, pieceW, pieceH)
    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)'
    ctx.fill()
    ctx.setLineDash([4, 3])
    ctx.strokeStyle = 'rgba(255,255,255,0.65)'
    ctx.lineWidth = 1.4
    ctx.stroke()
    ctx.restore()
  }, [sceneIndex, gapX])

  useEffect(() => {
    const canvas = pieceCanvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, pieceW + 4, pieceH + 4)

    ctx.save()
    puzzlePiecePath(ctx, 2, 2, pieceW, pieceH)
    ctx.clip()

    const offscreen = document.createElement('canvas')
    offscreen.width = canvasW
    offscreen.height = canvasH
    const offCtx = offscreen.getContext('2d')
    drawScene(offCtx, canvasW, canvasH, sceneIndex)

    const gapPxX = (gapX / 100) * canvasW
    ctx.drawImage(offscreen, -gapPxX + 2, -pieceY + 2)
    ctx.restore()

    ctx.save()
    puzzlePiecePath(ctx, 2, 2, pieceW, pieceH)
    ctx.strokeStyle = 'rgba(255,255,255,0.8)'
    ctx.lineWidth = 1.4
    ctx.stroke()
    ctx.restore()
  }, [sceneIndex, gapX])

  const getSliderPct = useCallback((e) => {
    if (!trackRef.current) return 0
    const rect = trackRef.current.getBoundingClientRect()
    const clientX = e.touches ? e.touches[0].clientX : e.clientX
    return Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100))
  }, [])

  const handleStart = useCallback((e) => {
    e.preventDefault()
    setDragging(true)
    setFeedback('')
    setSliderPct(getSliderPct(e))
  }, [getSliderPct])

  const handleMove = useCallback((e) => {
    if (!dragging) return
    e.preventDefault()
    setSliderPct(getSliderPct(e))
  }, [dragging, getSliderPct])

  const handleEnd = useCallback(() => {
    if (!dragging) return
    setDragging(false)

    if (Math.abs(sliderPct - gapX) <= 5) {
      setSolved(true)
      setFeedback('Verified!')
      setTimeout(() => onComplete(true), 250)
      return
    }

    const nextAttempts = attempts + 1
    setAttempts(nextAttempts)

    if (nextAttempts >= maxAttempts) {
      onComplete(false)
    } else {
      setFeedback(`Not quite. ${maxAttempts - nextAttempts} attempt(s) left.`)
      setSliderPct(0)
    }
  }, [dragging, sliderPct, gapX, attempts, maxAttempts, onComplete])

  useEffect(() => {
    if (!dragging) return
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
  }, [dragging, handleMove, handleEnd])

  const piecePxOffset = (sliderPct / 100) * canvasW

  return (
    <div className="challenge-inner">
      <p className="challenge-instruction">Drag the piece to fill the gap in the image.</p>

      <div className="jigsaw-scene-wrap">
        <canvas
          ref={bgCanvasRef}
          width={canvasW}
          height={canvasH}
          className="jigsaw-bg-canvas"
        />
        <canvas
          ref={pieceCanvasRef}
          width={pieceW + 4}
          height={pieceH + 4}
          className={`jigsaw-piece-canvas ${solved ? 'solved' : ''}`}
          style={{
            left: `${piecePxOffset}px`,
            top: `${pieceY - 2}px`,
          }}
        />
      </div>

      <div
        className="jigsaw-slider-track"
        ref={trackRef}
        onMouseDown={handleStart}
        onTouchStart={handleStart}
      >
        <div className="jigsaw-slider-rail" />
        <div className="jigsaw-slider-handle" style={{ left: `${sliderPct}%` }}>
          <span className="jigsaw-handle-arrow">⟷</span>
        </div>
      </div>

      {feedback && (
        <p className={`attempt-message ${feedback === 'Verified!' ? 'success' : ''}`}>
          {feedback}
        </p>
      )}
    </div>
  )
}

/* HARD: Moving Sequence Click Challenge
 Click three moving targets in order: 1, then 2, then 3.
 This keeps the game-like interaction from the older hard challenge, but with
 lower friction than a more complex multi-step puzzle. Target positions update
 in a canvas animation loop so motion is smooth and independent of React's
 render cadence. */

function makeTargets() {
  const colors = ['#5b8def', '#22c55e', '#f97316']
  return [1, 2, 3].map((label, index) => ({
    label,
    x: 72 + Math.random() * 256,
    y: 74 + Math.random() * 122,
    vx: (Math.random() > 0.5 ? 1 : -1) * (0.16 + Math.random() * 0.14),
    vy: (Math.random() > 0.5 ? 1 : -1) * (0.14 + Math.random() * 0.12),
    radius: 24,
    clicked: false,
    color: colors[index],
  }))
}

function drawMovingTargets(ctx, width, height, targets, nextExpected, timeLeft, active) {
  ctx.clearRect(0, 0, width, height)
  ctx.fillStyle = '#f8fafc'
  ctx.fillRect(0, 0, width, height)
  ctx.strokeStyle = '#d7deea'
  ctx.lineWidth = 2
  ctx.strokeRect(1, 1, width - 2, height - 2)

  targets.forEach((target) => {
    ctx.beginPath()
    ctx.arc(target.x, target.y, target.radius, 0, Math.PI * 2)
    ctx.fillStyle = target.clicked ? '#d5dce7' : target.color
    ctx.fill()
    ctx.strokeStyle = target.clicked ? '#8ea0b7' : '#1e293b'
    ctx.lineWidth = 2
    ctx.stroke()

    ctx.fillStyle = target.clicked ? '#475569' : '#ffffff'
    ctx.font = 'bold 20px Segoe UI'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(String(target.label), target.x, target.y)
  })

  ctx.fillStyle = '#0f172a'
  ctx.font = '600 15px Segoe UI'
  ctx.textAlign = 'left'
  ctx.fillText(`Next: ${nextExpected}`, 12, 22)

  ctx.textAlign = 'right'
  ctx.fillStyle = timeLeft <= 3 ? '#dc2626' : '#0f172a'
  ctx.fillText(`${timeLeft.toFixed(1)}s`, width - 12, 22)

  if (!active) {
    ctx.fillStyle = 'rgba(15, 23, 42, 0.08)'
    ctx.fillRect(0, 0, width, height)
  }
}

function MovingSequenceClickChallenge({ onComplete }) {
  const canvasRef = useRef(null)
  const animationRef = useRef(null)
  const timerRef = useRef(null)
  const lastTimeRef = useRef(0)
  const targetsRef = useRef(makeTargets())
  const nextExpectedRef = useRef(1)
  const timeLeftRef = useRef(9)
  const activeRef = useRef(true)

  const [nextExpected, setNextExpected] = useState(1)
  const [timeLeft, setTimeLeft] = useState(12)
  const [active, setActive] = useState(true)
  const [attempts, setAttempts] = useState(0)
  const [feedback, setFeedback] = useState('')

  const maxAttempts = 2
  const width = 400
  const height = 250

  useEffect(() => {
    nextExpectedRef.current = nextExpected
  }, [nextExpected])

  useEffect(() => {
    timeLeftRef.current = timeLeft
  }, [timeLeft])

  useEffect(() => {
    activeRef.current = active
  }, [active])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')

    const render = () => {
      drawMovingTargets(
        ctx,
        width,
        height,
        targetsRef.current,
        nextExpectedRef.current,
        timeLeftRef.current,
        activeRef.current
      )
    }

    render()

    if (!active) return

    const animate = (now) => {
      if (!lastTimeRef.current) lastTimeRef.current = now
      const dt = Math.min(40, now - lastTimeRef.current) / 16.67
      lastTimeRef.current = now

      targetsRef.current = targetsRef.current.map((target) => {
        if (target.clicked) return target

        let x = target.x + target.vx * dt
        let y = target.y + target.vy * dt
        let vx = target.vx
        let vy = target.vy

        if (x <= target.radius || x >= width - target.radius) vx *= -1
        if (y <= 36 + target.radius || y >= height - target.radius) vy *= -1

        x = Math.max(target.radius, Math.min(width - target.radius, x))
        y = Math.max(36 + target.radius, Math.min(height - target.radius, y))

        return { ...target, x, y, vx, vy }
      })

      render()
      animationRef.current = requestAnimationFrame(animate)
    }

    animationRef.current = requestAnimationFrame(animate)
    timerRef.current = window.setInterval(() => {
      setTimeLeft((prev) => Math.max(0, prev - 0.05))
    }, 50)

    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
      if (timerRef.current) window.clearInterval(timerRef.current)
      lastTimeRef.current = 0
    }
  }, [active])

  useEffect(() => {
    if (!active || timeLeft > 0) return

    setActive(false)
    const nextAttempts = attempts + 1
    setAttempts(nextAttempts)

    if (nextAttempts >= maxAttempts) {
      onComplete(false)
    } else {
      setFeedback(`Time ran out. ${maxAttempts - nextAttempts} retry left.`)
    }
  }, [timeLeft, active, attempts, maxAttempts, onComplete])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    drawMovingTargets(ctx, width, height, targetsRef.current, nextExpected, timeLeft, active)
  }, [nextExpected, timeLeft, active])

  const handleCanvasClick = (e) => {
    if (!active || !canvasRef.current) return

    const rect = canvasRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    for (const target of targetsRef.current) {
      const dist = Math.hypot(target.x - x, target.y - y)

      if (dist <= target.radius + 4) {
        if (target.label === nextExpected) {
          targetsRef.current = targetsRef.current.map((item) =>
            item.label === target.label ? { ...item, clicked: true } : item
          )

          if (nextExpected === 3) {
            setActive(false)
            setFeedback('Verified!')
            setTimeout(() => onComplete(true), 250)
          } else {
            setNextExpected((prev) => prev + 1)
            setFeedback('')
          }
        } else {
          setFeedback('Wrong order. Start again from 1.')
          targetsRef.current = targetsRef.current.map((item) => ({
            ...item,
            clicked: false,
          }))
          setNextExpected(1)
        }
        break
      }
    }
  }

  const handleRetry = () => {
    targetsRef.current = makeTargets()
    setNextExpected(1)
    setTimeLeft(12)
    setActive(true)
    setFeedback('')
  }

  return (
    <div className="challenge-inner">
      <p className="challenge-instruction">
        Click the moving circles in order: 1, then 2, then 3.
      </p>

      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="moving-click-canvas"
        onClick={handleCanvasClick}
      />

      {feedback && (
        <p className={`attempt-message ${feedback === 'Verified!' ? 'success' : ''}`}>
          {feedback}
        </p>
      )}

      {!active && attempts < maxAttempts && feedback !== 'Verified!' && (
        <button type="button" className="challenge-btn" onClick={handleRetry}>
          Try Again
        </button>
      )}
    </div>
  )
}

function ChallengeModal({ type, difficulty, onComplete }) {
  if (type === 'blocked') {
    return (
      <div className="challenge-overlay">
        <div className="challenge-modal">
          <h2>Access Denied</h2>
          <p className="challenge-instruction">
            Our system has detected unusual activity. This checkout has been blocked.
          </p>
          <button type="button" className="challenge-btn" onClick={() => onComplete(false)}>
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
        {difficulty === 'easy' && <RotationChallenge onComplete={onComplete} />}
        {difficulty === 'medium' && <JigsawSliderChallenge onComplete={onComplete} />}
        {difficulty === 'hard' && <MovingSequenceClickChallenge onComplete={onComplete} />}
        {!['easy', 'medium', 'hard'].includes(difficulty) && (
          <RotationChallenge onComplete={onComplete} />
        )}
      </div>
    </div>
  )
}

export default ChallengeModal