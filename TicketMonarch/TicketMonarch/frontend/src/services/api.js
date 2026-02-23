// Backend API base URL
// In development we proxy /api to Flask; use a relative base path.
const API_BASE_URL = '/api'

export const submitCheckout = async (checkoutData) => {
  try {
    const response = await fetch(`${API_BASE_URL}/checkout`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(checkoutData),
    })

    const data = await response.json()

    if (response.ok && data.success) {
      return { success: true }
    }

    return { success: false }

  } catch (error) {
    return { success: false }
  }
}

/**
 * Get all orders from the backend
 * @returns {Promise<Object>} Response object with orders data
 */
/**
 * Export checkout data to CSV
 * @returns {Promise<Object>} Response object with export status
 */
export const exportCheckouts = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/export`)
    const data = await response.json()

    if (response.ok && data.success) {
      return {
        success: true,
        message: data.message || 'Data exported successfully',
        filePath: data.file_path
      }
    }

    return {
      success: false,
      error: data.error || 'Failed to export data'
    }
  } catch (error) {
    console.error('Export API error:', error)
    return {
      success: false,
      error: 'Network error',
      message: 'Unable to export data. Please check your connection.'
    }
  }
}

/**
 * Health check endpoint
 * @returns {Promise<Object>} Response object with health status
 */
export const healthCheck = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/health`)
    const data = await response.json()

    return {
      success: response.ok,
      data: data
    }
  } catch (error) {
    console.error('Health check API error:', error)
    return {
      success: false,
      error: 'Unable to connect to the server'
    }
  }
}

export const evaluateSession = async (sessionId) => {
  try {
    const response = await fetch(`${API_BASE_URL}/agent/evaluate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId }),
    })
    return await response.json()
  } catch (error) {
    console.error('Agent evaluation error:', error)
    return { success: true, decision: 'allow', action_index: 5, reason: 'agent_unreachable' }
  }
}

export const fetchDashboardData = async (sessionId) => {
  try {
    const response = await fetch(`${API_BASE_URL}/agent/dashboard/${sessionId}`)
    return await response.json()
  } catch (error) {
    return { success: false, error: 'Network error' }
  }
}

export const fetchRecentSessions = async (limit = 20) => {
  try {
    const response = await fetch(`${API_BASE_URL}/agent/sessions?limit=${limit}`)
    return await response.json()
  } catch (error) {
    return { success: false, sessions: [] }
  }
}

export const fetchLiveTelemetry = async (sessionId) => {
  try {
    const response = await fetch(`${API_BASE_URL}/agent/live/${sessionId}`)
    return await response.json()
  } catch (error) {
    return { success: false }
  }
}

