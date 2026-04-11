import { describe, test, expect, vi, beforeEach, afterEach } from 'vitest'

vi.mock('uuid', () => ({
  v4: vi.fn(),
}))

describe('tracking service', () => {
  let getSessionId
  let setTrackingPage
  let initTracking
  let forceFlush
  let resetSession
  let uuidv4

  beforeEach(async () => {
    vi.resetModules()
    vi.clearAllMocks()
    vi.useFakeTimers()

    sessionStorage.clear()
    localStorage.clear()

    global.fetch = vi.fn().mockResolvedValue({ ok: true })

    vi.spyOn(console, 'log').mockImplementation(() => {})
    vi.spyOn(console, 'error').mockImplementation(() => {})

    const uuidModule = await import('uuid')
    uuidv4 = uuidModule.v4
    uuidv4.mockReturnValue('mock-uuid-1')

    const tracking = await import('@/services/tracking')
    getSessionId = tracking.getSessionId
    setTrackingPage = tracking.setTrackingPage
    initTracking = tracking.initTracking
    forceFlush = tracking.forceFlush
    resetSession = tracking.resetSession
  })

  afterEach(() => {
    vi.runOnlyPendingTimers()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  test('getSessionId returns existing sessionStorage value when present', () => {
    sessionStorage.setItem('tm_session_id', 'existing-session')

    const id = getSessionId()

    expect(id).toBe('existing-session')
    expect(sessionStorage.getItem('tm_session_id')).toBe('existing-session')
  })

  test('getSessionId creates and stores a new session id when absent', () => {
    const id = getSessionId()

    expect(id).toBe('mock-uuid-1')
    expect(sessionStorage.getItem('tm_session_id')).toBe('mock-uuid-1')
    expect(localStorage.getItem('tm_active_session_id')).toBe('mock-uuid-1')
  })

  test('initTracking registers listeners and initializes session', () => {
    const addEventListenerSpy = vi.spyOn(window, 'addEventListener')

    initTracking()

    expect(addEventListenerSpy).toHaveBeenCalledWith(
      'mousemove',
      expect.any(Function),
      { passive: true }
    )
    expect(addEventListenerSpy).toHaveBeenCalledWith(
      'click',
      expect.any(Function),
      { passive: true }
    )
    expect(addEventListenerSpy).toHaveBeenCalledWith(
      'wheel',
      expect.any(Function),
      { passive: true }
    )

    expect(sessionStorage.getItem('tm_session_id')).toBe('mock-uuid-1')
  })

  test('forceFlush sends buffered click telemetry after interaction', async () => {
    initTracking()
    setTrackingPage('checkout')

    const button = document.createElement('button')
    button.id = 'buy-btn'
    button.className = 'primary'
    button.name = 'buy'
    button.type = 'button'
    button.innerText = 'Buy'

    const clickEvent = new MouseEvent('click', {
      bubbles: true,
      clientX: 50,
      clientY: 80,
      button: 0,
    })

    Object.defineProperty(clickEvent, 'target', {
      value: button,
      enumerable: true,
    })

    window.dispatchEvent(clickEvent)

    await forceFlush()

    expect(fetch).toHaveBeenCalledWith(
      '/api/tracking/clicks',
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      })
    )

    const [, options] = fetch.mock.calls[0]
    const body = JSON.parse(options.body)

    expect(body.session_id).toBe('mock-uuid-1')
    expect(body.page).toBe('checkout')
    expect(body.clicks).toHaveLength(1)
    expect(body.clicks[0].button).toBe('left')
  })

  test('resetSession creates a new session id and clears bot flag', () => {
    sessionStorage.setItem('tm_session_id', 'old-session')
    sessionStorage.setItem('tm_is_bot', 'true')
    localStorage.setItem('tm_active_session_id', 'old-session')

    uuidv4.mockReturnValue('new-session')

    resetSession()

    expect(sessionStorage.getItem('tm_session_id')).toBe('new-session')
    expect(sessionStorage.getItem('tm_is_bot')).toBeNull()
    expect(localStorage.getItem('tm_active_session_id')).toBe('new-session')
    expect(console.log).toHaveBeenCalled()
  })

  test('initTracking does nothing on /dev path', async () => {
    window.history.pushState({}, '', '/dev/dashboard')

    const addEventListenerSpy = vi.spyOn(window, 'addEventListener')

    initTracking()

    expect(addEventListenerSpy).not.toHaveBeenCalledWith(
      'mousemove',
      expect.any(Function),
      { passive: true }
    )
    expect(sessionStorage.getItem('tm_session_id')).toBeNull()
  })

  test('forceFlush sends mouse samples after mousemove and sampling interval', async () => {
    window.history.pushState({}, '', '/checkout')
    initTracking()
    setTrackingPage('checkout')

    window.dispatchEvent(new MouseEvent('mousemove', {
      clientX: 10,
      clientY: 20,
    }))

    vi.advanceTimersByTime(20)
    await forceFlush()

    expect(fetch).toHaveBeenCalledWith(
      '/api/tracking/mouse',
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      })
    )

    const mouseCall = fetch.mock.calls.find(([url]) => url === '/api/tracking/mouse')
    const body = JSON.parse(mouseCall[1].body)

    expect(body.samples).toHaveLength(1)
    expect(body.samples[0].x).toBe(10)
    expect(body.samples[0].y).toBe(20)
  })

  test('forceFlush sends scroll payload after wheel event', async () => {
    window.history.pushState({}, '', '/checkout')
    initTracking()
    setTrackingPage('checkout')

    window.dispatchEvent(new WheelEvent('wheel', {
      deltaY: 120,
    }))

    await forceFlush()

    const scrollCall = fetch.mock.calls.find(([url]) => url === '/api/tracking/scroll')
    expect(scrollCall).toBeTruthy()

    const body = JSON.parse(scrollCall[1].body)
    expect(body.scrolls).toHaveLength(1)
    expect(body.scrolls[0].dy).toBe(120)
  })

  test('forceFlush sends keystrokes for form field special keys', async () => {
    window.history.pushState({}, '', '/checkout')
    initTracking()
    setTrackingPage('checkout')

    const input = document.createElement('input')
    input.name = 'email'
    document.body.appendChild(input)

    const down = new KeyboardEvent('keydown', { key: 'Enter', bubbles: true })
    const up = new KeyboardEvent('keyup', { key: 'Enter', bubbles: true })

    Object.defineProperty(down, 'target', { value: input })
    Object.defineProperty(up, 'target', { value: input })

    window.dispatchEvent(down)
    window.dispatchEvent(up)

    await forceFlush()

    const keyCall = fetch.mock.calls.find(([url]) => url === '/api/tracking/keystrokes')
    expect(keyCall).toBeTruthy()

    const body = JSON.parse(keyCall[1].body)
    expect(body.keystrokes).toHaveLength(2)
    expect(body.keystrokes[0].key).toBe('Enter')
    expect(body.keystrokes[1].key).toBe('Enter')
  })

  test('regular keys are recorded without exposing the key value', async () => {
    window.history.pushState({}, '', '/checkout')
    initTracking()
    setTrackingPage('checkout')

    const input = document.createElement('input')
    input.name = 'email'
    document.body.appendChild(input)

    const down = new KeyboardEvent('keydown', { key: 'a', bubbles: true })
    Object.defineProperty(down, 'target', { value: input })

    window.dispatchEvent(down)
    await forceFlush()

    const keyCall = fetch.mock.calls.find(([url]) => url === '/api/tracking/keystrokes')
    const body = JSON.parse(keyCall[1].body)

    expect(body.keystrokes[0].field).toBe('email')
    expect(body.keystrokes[0].key).toBeUndefined()
  })

  test('setTrackingPage(null) disables tracking', async () => {
    window.history.pushState({}, '', '/checkout')
    initTracking()
    setTrackingPage(null)

    window.dispatchEvent(new MouseEvent('click', {
      clientX: 1,
      clientY: 2,
      button: 0,
    }))

    await forceFlush()

    expect(fetch).not.toHaveBeenCalled()
  })
})