import { describe, test, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, act, cleanup } from '@testing-library/react'
import ChallengeModal from '@/components/ChallengeModal'

const mockCtx = {
  clearRect: vi.fn(),
  save: vi.fn(),
  restore: vi.fn(),
  translate: vi.fn(),
  rotate: vi.fn(),
  fillRect: vi.fn(),
  beginPath: vi.fn(),
  moveTo: vi.fn(),
  lineTo: vi.fn(),
  closePath: vi.fn(),
  fill: vi.fn(),
  stroke: vi.fn(),
  arc: vi.fn(),
  clip: vi.fn(),
  drawImage: vi.fn(),
  setLineDash: vi.fn(),
  strokeRect: vi.fn(),
  fillText: vi.fn(),
  createLinearGradient: vi.fn(() => ({
    addColorStop: vi.fn(),
  })),
}

function mockHardChallengeRandoms() {
  return vi.spyOn(Math, 'random').mockReturnValue(0)
}

let rafSpy
let cafSpy

describe('ChallengeModal', () => {
  let originalCreateElement

  beforeEach(() => {
    vi.useFakeTimers()

    rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation(() => 1)
    cafSpy = vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => {})

    HTMLCanvasElement.prototype.getContext = vi.fn(() => mockCtx)

    originalCreateElement = document.createElement.bind(document)
    vi.spyOn(document, 'createElement').mockImplementation((tagName, options) => {
      const el = originalCreateElement(tagName, options)
      if (tagName === 'canvas') {
        el.getContext = vi.fn(() => mockCtx)
      }
      return el
    })
  })

  afterEach(() => {
    cleanup()
    vi.runOnlyPendingTimers()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  test('renders blocked state and calls onComplete(false) when Go Back is clicked', () => {
    const onComplete = vi.fn()

    render(
      <ChallengeModal
        type="blocked"
        difficulty="easy"
        onComplete={onComplete}
      />
    )

    expect(screen.getByText(/Access Denied/i)).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /Go Back/i }))

    expect(onComplete).toHaveBeenCalledWith(false)
  })

  test('renders easy challenge by default', () => {
    const onComplete = vi.fn()

    render(
      <ChallengeModal
        type="challenge"
        difficulty="easy"
        onComplete={onComplete}
      />
    )

    expect(screen.getByText(/Rotate the object until it is upright/i)).toBeInTheDocument()
    expect(screen.getByRole('slider', { name: /Rotation angle/i })).toBeInTheDocument()
  })

  test('renders medium challenge', () => {
    const onComplete = vi.fn()

    render(
      <ChallengeModal
        type="challenge"
        difficulty="medium"
        onComplete={onComplete}
      />
    )

    expect(screen.getByText(/Drag the piece to fill the gap in the image/i)).toBeInTheDocument()
  })

  test('renders hard challenge', () => {
    const onComplete = vi.fn()

    act(() => {
      render(
        <ChallengeModal
          type="challenge"
          difficulty="hard"
          onComplete={onComplete}
        />
      )
    })

    expect(
      screen.getByText(/Click the moving circles in order/i)
    ).toBeInTheDocument()
  })

  test('falls back to easy challenge for unknown difficulty', () => {
    const onComplete = vi.fn()

    render(
      <ChallengeModal
        type="challenge"
        difficulty="unknown"
        onComplete={onComplete}
      />
    )

    expect(screen.getByText(/Rotate the object until it is upright/i)).toBeInTheDocument()
  })

  test('easy challenge shows retry feedback on wrong submit and fails after 3 attempts', async () => {
    const onComplete = vi.fn()

    // objectIndex + baseAngle setup
    vi.spyOn(Math, 'random')
      .mockReturnValueOnce(0)    // objectIndex
      .mockReturnValueOnce(0)    // angle => 40
      .mockReturnValueOnce(0.9)  // choose positive angle, so baseAngle = 40

    render(
      <ChallengeModal
        type="challenge"
        difficulty="easy"
        onComplete={onComplete}
      />
    )

    const submit = screen.getByRole('button', { name: /submit/i })

    fireEvent.click(submit)
    expect(screen.getByText(/2 attempt\(s\) left/i)).toBeInTheDocument()

    fireEvent.click(submit)
    expect(screen.getByText(/1 attempt\(s\) left/i)).toBeInTheDocument()

    fireEvent.click(submit)
    expect(onComplete).toHaveBeenCalledWith(false)
  })

  test('easy challenge succeeds when rotated upright', () => {
    const onComplete = vi.fn()

    vi.spyOn(Math, 'random')
      .mockReturnValueOnce(0)    // objectIndex
      .mockReturnValueOnce(0)    // angle => 40
      .mockReturnValueOnce(0.9)  // baseAngle = 40

    render(
      <ChallengeModal
        type="challenge"
        difficulty="easy"
        onComplete={onComplete}
      />
    )

    const slider = screen.getByRole('slider', { name: /rotation angle/i })
    fireEvent.change(slider, { target: { value: '40' } })

    fireEvent.click(screen.getByRole('button', { name: /submit/i }))

    expect(screen.getByText(/Verified!/i)).toBeInTheDocument()

    vi.advanceTimersByTime(250)
    expect(onComplete).toHaveBeenCalledWith(true)
  })

  test('easy challenge rotate buttons work', () => {
    const onComplete = vi.fn()

    vi.spyOn(Math, 'random')
      .mockReturnValueOnce(0)
      .mockReturnValueOnce(0)
      .mockReturnValueOnce(0.9)

    render(
      <ChallengeModal
        type="challenge"
        difficulty="easy"
        onComplete={onComplete}
      />
    )

    fireEvent.click(screen.getByRole('button', { name: /submit/i }))
    expect(screen.getByText(/2 attempt\(s\) left/i)).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /Rotate clockwise/i }))
    expect(screen.queryByText(/2 attempt\(s\) left/i)).not.toBeInTheDocument()
  })

  test('medium challenge succeeds when piece is dragged near the gap', () => {
    const onComplete = vi.fn()

    vi.spyOn(Math, 'random')
      .mockReturnValueOnce(0) // sceneIndex
      .mockReturnValueOnce(0) // gapX => 25

    render(
      <ChallengeModal
        type="challenge"
        difficulty="medium"
        onComplete={onComplete}
      />
    )

    const track = document.querySelector('.jigsaw-slider-track')
    vi.spyOn(track, 'getBoundingClientRect').mockReturnValue({
      left: 0,
      width: 100,
      top: 0,
      right: 100,
      bottom: 0,
      height: 0,
      x: 0,
      y: 0,
      toJSON: () => {},
    })

    fireEvent.mouseDown(track, { clientX: 25 })
    fireEvent.mouseMove(window, { clientX: 25 })
    fireEvent.mouseUp(window)

    expect(screen.getByText(/Verified!/i)).toBeInTheDocument()

    vi.advanceTimersByTime(250)
    expect(onComplete).toHaveBeenCalledWith(true)
  })

  test('medium challenge shows retry feedback and fails after max attempts', () => {
    const onComplete = vi.fn()

    vi.spyOn(Math, 'random')
      .mockReturnValueOnce(0) // sceneIndex
      .mockReturnValueOnce(0) // gapX => 25

    render(
      <ChallengeModal
        type="challenge"
        difficulty="medium"
        onComplete={onComplete}
      />
    )

    const track = document.querySelector('.jigsaw-slider-track')
    vi.spyOn(track, 'getBoundingClientRect').mockReturnValue({
      left: 0,
      width: 100,
      top: 0,
      right: 100,
      bottom: 0,
      height: 0,
      x: 0,
      y: 0,
      toJSON: () => {},
    })

    const miss = () => {
      fireEvent.mouseDown(track, { clientX: 90 })
      fireEvent.mouseMove(window, { clientX: 90 })
      fireEvent.mouseUp(window)
    }

    miss()
    expect(screen.getByText(/2 attempt\(s\) left/i)).toBeInTheDocument()

    miss()
    expect(screen.getByText(/1 attempt\(s\) left/i)).toBeInTheDocument()

    miss()
    expect(onComplete).toHaveBeenCalledWith(false)
  })

  test('hard challenge shows timeout feedback and retry button', () => {
    const onComplete = vi.fn()

    act(() => {
      render(
        <ChallengeModal
          type="challenge"
          difficulty="hard"
          onComplete={onComplete}
        />
      )
    })

    act(() => {
      vi.advanceTimersByTime(12100)
    })

    expect(screen.getByText(/Time ran out\./i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Try Again/i })).toBeInTheDocument()
    expect(onComplete).not.toHaveBeenCalled()
  })

  test('hard challenge fails after second timeout', () => {
    const onComplete = vi.fn()

    act(() => {
      render(
        <ChallengeModal
          type="challenge"
          difficulty="hard"
          onComplete={onComplete}
        />
      )
    })

    act(() => {
      vi.advanceTimersByTime(12100)
    })

    act(() => {
      fireEvent.click(screen.getByRole('button', { name: /Try Again/i }))
    })

    act(() => {
      vi.advanceTimersByTime(12100)
    })

    expect(onComplete).toHaveBeenCalledWith(false)
  })

  test('hard challenge shows wrong order feedback when clicking out of sequence', () => {
    const onComplete = vi.fn()

    vi.spyOn(Math, 'random').mockReturnValue(0.1)
    vi.spyOn(Math, 'hypot')
      .mockReturnValueOnce(100) // target 1 miss
      .mockReturnValueOnce(1)   // target 2 hit

    act(() => {
      render(
        <ChallengeModal
          type="challenge"
          difficulty="hard"
          onComplete={onComplete}
        />
      )
    })

    const canvas = document.querySelector('.moving-click-canvas')
    vi.spyOn(canvas, 'getBoundingClientRect').mockReturnValue({
      left: 0,
      top: 0,
      width: 400,
      height: 250,
      right: 400,
      bottom: 250,
      x: 0,
      y: 0,
      toJSON: () => {},
    })

    act(() => {
      fireEvent.click(canvas, { clientX: 100, clientY: 100 })
    })

    expect(screen.getByText(/Wrong order\. Start again from 1\./i)).toBeInTheDocument()
    expect(onComplete).not.toHaveBeenCalled()
  })

  test.each([
    [0.30, 'variant 2'],
    [0.60, 'variant 3'],
    [0.90, 'variant 4'],
  ])('easy challenge renders alternate object %s (%s)', (randomValue) => {
    const onComplete = vi.fn()

    vi.spyOn(Math, 'random')
      .mockReturnValueOnce(randomValue) // objectIndex
      .mockReturnValueOnce(0)           // angle => 40
      .mockReturnValueOnce(0.9)         // positive angle

    render(
      <ChallengeModal
        type="challenge"
        difficulty="easy"
        onComplete={onComplete}
      />
    )

    expect(
      screen.getByText(/Rotate the object until it is upright/i)
    ).toBeInTheDocument()

    expect(screen.getByRole('slider', { name: /Rotation angle/i })).toBeInTheDocument()
  })

  test.each([
    [0.4, 'scene 2'],
    [0.8, 'scene 3'],
  ])('medium challenge renders alternate scene %s (%s)', (sceneRandom) => {
    const onComplete = vi.fn()

    vi.spyOn(Math, 'random')
      .mockReturnValueOnce(sceneRandom) // sceneIndex
      .mockReturnValueOnce(0)           // gapX => 25

    render(
      <ChallengeModal
        type="challenge"
        difficulty="medium"
        onComplete={onComplete}
      />
    )

    expect(
      screen.getByText(/Drag the piece to fill the gap in the image/i)
    ).toBeInTheDocument()
  })
})