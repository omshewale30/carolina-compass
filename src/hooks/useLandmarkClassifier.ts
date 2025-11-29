import { useCallback, useEffect, useState } from 'react'

export type ModelStatus = 'loading' | 'ready' | 'error'

// API endpoint - can be overridden with environment variable
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const useLandmarkClassifier = () => {
  const [status, setStatus] = useState<ModelStatus>('loading')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let isMounted = true

    const checkHealth = async () => {
      try {
        const response = await fetch(`${API_URL}/health`)
        if (!response.ok) {
          throw new Error('API health check failed')
        }
        const data = await response.json()
        if (!isMounted) return
        
        if (data.model_ready) {
          setStatus('ready')
        } else {
          setStatus('error')
          setError('Model is not ready on the server.')
        }
      } catch (err) {
        console.error('Failed to connect to API', err)
        if (!isMounted) return
        setError('Unable to connect to the inference server. Please ensure the backend is running.')
        setStatus('error')
      }
    }

    checkHealth()
    
    // Poll health check every 5 seconds if not ready
    const interval = setInterval(() => {
      if (status === 'loading' || status === 'error') {
        checkHealth()
      }
    }, 5000)

    return () => {
      isMounted = false
      clearInterval(interval)
    }
  }, [status])

  const predict = useCallback(async (file: File): Promise<number[]> => {
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }))
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      return data.predictions as number[]
    } catch (err) {
      console.error('Prediction error', err)
      throw err
    }
  }, [])

  return {
    predict,
    status,
    error,
  }
}

export default useLandmarkClassifier

