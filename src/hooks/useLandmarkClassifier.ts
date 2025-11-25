import { useCallback, useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'

export type ModelStatus = 'loading' | 'ready' | 'error'

const MODEL_URL = '/model_web/model.json'

const useLandmarkClassifier = () => {
  const modelRef = useRef<tf.GraphModel | null>(null)
  const [status, setStatus] = useState<ModelStatus>('loading')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let isMounted = true

    const loadModel = async () => {
      try {
        const model = await tf.loadGraphModel(MODEL_URL)
        if (!isMounted) return
        modelRef.current = model
        setStatus('ready')
      } catch (err) {
        console.error('Failed to load model', err)
        if (!isMounted) return
        setError('We had trouble loading the landmark model. Please try again.')
        setStatus('error')
      }
    }

    loadModel()

    return () => {
      isMounted = false
      modelRef.current?.dispose()
      modelRef.current = null
    }
  }, [])

  const predict = useCallback(async (input: tf.Tensor4D) => {
    const model = modelRef.current
    if (!model) {
      throw new Error('Model is not ready yet.')
    }
    const predictions = model.predict(input) as tf.Tensor
    const data = await predictions.data()
    predictions.dispose()
    return Array.from(data)
  }, [])

  return {
    predict,
    status,
    error,
  }
}

export default useLandmarkClassifier

