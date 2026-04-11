import { describe, test, expect, vi, beforeEach, afterEach } from 'vitest'
import {
  submitCheckout,
  exportCheckouts,
  healthCheck,
  setFlag,
  getFlag,
  rollingEvaluate,
  evaluateSession,
  fetchDashboardData,
  fetchRecentSessions,
  fetchLiveTelemetry,
  confirmHumanSession,
} from '@/services/api'

describe('api service', () => {
  beforeEach(() => {
    global.fetch = vi.fn()
    vi.spyOn(console, 'error').mockImplementation(() => {})
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  test('submitCheckout returns success true on ok response with success true', async () => {
    fetch.mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue({ success: true }),
    })

    const result = await submitCheckout({ name: 'Josh' })

    expect(fetch).toHaveBeenCalledWith('/api/checkout', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: 'Josh' }),
    })

    expect(result).toEqual({ success: true })
  })

  test('submitCheckout returns success false on non-success payload', async () => {
    fetch.mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue({ success: false }),
    })

    const result = await submitCheckout({})

    expect(result).toEqual({ success: false })
  })

  test('submitCheckout returns success false on fetch error', async () => {
    fetch.mockRejectedValue(new Error('network error'))

    const result = await submitCheckout({})

    expect(result).toEqual({ success: false })
  })

  test('exportCheckouts returns mapped success response', async () => {
    fetch.mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue({
        success: true,
        message: 'done',
        file_path: '/tmp/export.csv',
      }),
    })

    const result = await exportCheckouts()

    expect(result).toEqual({
      success: true,
      message: 'done',
      filePath: '/tmp/export.csv',
    })
  })

  test('exportCheckouts returns fallback error on network failure', async () => {
    fetch.mockRejectedValue(new Error('network error'))

    const result = await exportCheckouts()

    expect(result).toEqual({
      success: false,
      error: 'Network error',
      message: 'Unable to export data. Please check your connection.',
    })
  })

  test('healthCheck returns response status and data', async () => {
    fetch.mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue({ status: 'ok' }),
    })

    const result = await healthCheck()

    expect(result).toEqual({
      success: true,
      data: { status: 'ok' },
    })
  })

  test('setFlag posts flag value', async () => {
    fetch.mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue({ updated: true }),
    })

    const result = await setFlag('blocked')

    expect(fetch).toHaveBeenCalledWith('/api/set_flag', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ flag: 'blocked' }),
    })

    expect(result).toEqual({
      success: true,
      data: { updated: true },
    })
  })

  test('getFlag returns flag data', async () => {
    fetch.mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue({ flag: 'allowed' }),
    })

    const result = await getFlag()

    expect(fetch).toHaveBeenCalledWith('/api/get_flag')
    expect(result).toEqual({
      success: true,
      data: { flag: 'allowed' },
    })
  })

  test('rollingEvaluate returns API json', async () => {
    fetch.mockResolvedValue({
      json: vi.fn().mockResolvedValue({ success: true, score: 0.1 }),
    })

    const result = await rollingEvaluate('session-123')

    expect(fetch).toHaveBeenCalledWith('/api/agent/rolling', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: 'session-123' }),
    })

    expect(result).toEqual({ success: true, score: 0.1 })
  })

  test('rollingEvaluate returns safe fallback on fetch error', async () => {
    fetch.mockRejectedValue(new Error('network error'))

    const result = await rollingEvaluate('session-123')

    expect(result).toEqual({
      success: true,
      bot_probability: 0,
      deploy_honeypot: false,
      events_processed: 0,
      honeypot_triggered: false,
    })
  })

  test('evaluateSession returns fallback on fetch error', async () => {
    fetch.mockRejectedValue(new Error('network error'))

    const result = await evaluateSession('session-123')

    expect(result).toEqual({
      success: false,
      decision: 'allow',
      action_index: 5,
      reason: 'agent_unreachable',
    })
  })

  test('fetchDashboardData returns network error fallback', async () => {
    fetch.mockRejectedValue(new Error('network error'))

    const result = await fetchDashboardData('session-123')

    expect(result).toEqual({
      success: false,
      error: 'Network error',
    })
  })

  test('fetchRecentSessions returns empty sessions on fetch error', async () => {
    fetch.mockRejectedValue(new Error('network error'))

    const result = await fetchRecentSessions()

    expect(result).toEqual({
      success: false,
      sessions: [],
    })
  })

  test('fetchLiveTelemetry returns success false on fetch error', async () => {
    fetch.mockRejectedValue(new Error('network error'))

    const result = await fetchLiveTelemetry('session-123')

    expect(result).toEqual({
      success: false,
    })
  })

  test('confirmHumanSession posts expected payload', async () => {
    fetch.mockResolvedValue({
      json: vi.fn().mockResolvedValue({ success: true, updated: true }),
    })

    const result = await confirmHumanSession('session-123')

    expect(fetch).toHaveBeenCalledWith('/api/agent/confirm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: 'session-123', true_label: 1 }),
    })

    expect(result).toEqual({ success: true, updated: true })
  })

  test('confirmHumanSession returns success false on fetch error', async () => {
    fetch.mockRejectedValue(new Error('network error'))

    const result = await confirmHumanSession('session-123')

    expect(result).toEqual({ success: false })
  })

  test('exportCheckouts uses default error message when api omits error', async () => {
    fetch.mockResolvedValue({
      ok: false,
      json: vi.fn().mockResolvedValue({}),
    })

    const result = await exportCheckouts()

    expect(result).toEqual({
      success: false,
      error: 'Failed to export data',
    })
  })

  test('healthCheck returns fallback on fetch error', async () => {
    fetch.mockRejectedValue(new Error('boom'))

    const result = await healthCheck()

    expect(result).toEqual({
      success: false,
      error: 'Unable to connect to the server',
    })
  })

  test('setFlag returns fallback on fetch error', async () => {
    fetch.mockRejectedValue(new Error('boom'))

    const result = await setFlag('blocked')

    expect(result).toEqual({
      success: false,
      error: 'Unable to connect to the server',
    })
  })

  test('getFlag returns fallback on fetch error', async () => {
    fetch.mockRejectedValue(new Error('boom'))

    const result = await getFlag()

    expect(result).toEqual({
      success: false,
      error: 'Unable to connect to the server',
    })
  })

  test('fetchDashboardData returns api json on success', async () => {
    fetch.mockResolvedValue({
      json: vi.fn().mockResolvedValue({ success: true, session: { id: 'a1' } }),
    })

    const result = await fetchDashboardData('a1')

    expect(result).toEqual({ success: true, session: { id: 'a1' } })
  })

  test('fetchRecentSessions passes custom limit', async () => {
    fetch.mockResolvedValue({
      json: vi.fn().mockResolvedValue({ success: true, sessions: [] }),
    })

    await fetchRecentSessions(5)

    expect(fetch).toHaveBeenCalledWith('/api/agent/sessions?limit=5')
  })

  test('fetchLiveTelemetry returns api json on success', async () => {
    fetch.mockResolvedValue({
      json: vi.fn().mockResolvedValue({ success: true, events: [] }),
    })

    const result = await fetchLiveTelemetry('s1')

    expect(result).toEqual({ success: true, events: [] })
  })
})