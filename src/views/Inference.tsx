import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import useLandmarkClassifier from '../hooks/useLandmarkClassifier'
import { FALLBACK_LANDMARK, LANDMARKS } from '../data/landmarks'
import { prepareImageTensor } from '../utils/imageProcessor'
import type { ChangeEvent } from 'react'
import type { Landmark } from '../data/landmarks'

type InferenceProps = {
  onBack: () => void
}

type PredictionResult = Landmark & {
  confidence: number
  index: number
}

const cameraIcon = (
  <svg viewBox="0 0 24 24" className="h-6 w-6" aria-hidden="true">
    <path
      d="M9.5 6L8 8H5a2 2 0 00-2 2v7a2 2 0 002 2h14a2 2 0 002-2v-7a2 2 0 00-2-2h-3l-1.5-2h-5z"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <circle cx="12" cy="13" r="3.2" fill="none" stroke="currentColor" strokeWidth="1.8" />
  </svg>
)

const uploadIcon = (
  <svg viewBox="0 0 24 24" className="h-6 w-6" aria-hidden="true">
    <path
      d="M12 16V5m0 0l3.5 3.5M12 5L8.5 8.5M5 16v1.5A1.5 1.5 0 006.5 19h11a1.5 1.5 0 001.5-1.5V16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
)

const Inference = ({ onBack }: InferenceProps) => {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const cameraInputRef = useRef<HTMLInputElement>(null)
  const { predict, status, error } = useLandmarkClassifier()

  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [analysisState, setAnalysisState] = useState<'idle' | 'processing' | 'error' | 'complete'>('idle')
  const [statusMessage, setStatusMessage] = useState('Capture a landmark to get started.')
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl)
      }
    }
  }, [previewUrl])

  const analyzeImage = useCallback(
    async (file: File) => {
      if (!file.type.startsWith('image/')) {
        setStatusMessage('Please choose an image file.')
        return
      }
      if (status === 'loading') {
        setStatusMessage('The AI model is still warming up. One sec...')
        return
      }
      if (status === 'error') {
        setStatusMessage(error ?? 'Model unavailable right now.')
        return
      }

      setAnalysisState('processing')
      setStatusMessage('Analyzing landmark...')

      let tensor
      try {
        const processed = await prepareImageTensor(file)
        tensor = processed.tensor
        setPreviewUrl((prev) => {
          if (prev) URL.revokeObjectURL(prev)
          return processed.previewUrl
        })

        const scores = await predict(processed.tensor)
        const { index, confidence } = scores.reduce(
          (acc, score, idx) => {
            if (score > acc.confidence) {
              return { index: idx, confidence: score }
            }
            return acc
          },
          { index: 0, confidence: 0 },
        )

        const landmark = LANDMARKS[index] ?? FALLBACK_LANDMARK
        setPrediction({
          ...landmark,
          confidence: Number.isFinite(confidence) ? confidence : 0,
          index,
        })
        setAnalysisState('complete')
        setStatusMessage('Here is what we found.')
      } catch (err) {
        console.error('Inference error', err)
        setAnalysisState('error')
        setPrediction(null)
        setStatusMessage('We hit a snag while analyzing. Try another shot.')
      } finally {
        tensor?.dispose()
      }
    },
    [error, predict, status],
  )

  const handleFileChange = useCallback(
    async (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0]
      event.target.value = ''
      if (!file) return
      await analyzeImage(file)
    },
    [analyzeImage],
  )

  const modelBadge = useMemo(() => {
    switch (status) {
      case 'ready':
        return { label: 'Ready to explore', tone: 'text-emerald-600 bg-emerald-100' }
      case 'error':
        return { label: 'Model unavailable', tone: 'text-rose-600 bg-rose-100' }
      default:
        return { label: 'Warming up MobileNetV3', tone: 'text-slate-600 bg-slate-100' }
    }
  }, [status])

  const confidencePercent = prediction ? Math.round(prediction.confidence * 100) : 0

  return (
    <section className="relative min-h-screen bg-gradient-to-b from-white via-slate-50 to-white pb-[320px] text-slate-900 md:pb-12">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-6 pt-8 md:gap-8 md:pt-12">
        <header className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <button
            onClick={onBack}
            className="inline-flex items-center gap-2 text-sm font-semibold text-[#4B9CD3] transition hover:text-[#2c6b94]"
          >
            <span aria-hidden="true">←</span> Back to welcome
          </button>
          <div className="flex flex-col items-start gap-1 text-sm text-slate-500 md:items-end">
            <span className="font-medium uppercase tracking-[0.2em] text-slate-400">Model status</span>
            <span className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-semibold ${modelBadge.tone}`}>
              <span className="relative flex h-2 w-2">
                <span
                  className={`absolute inline-flex h-full w-full rounded-full opacity-75 ${status === 'ready' ? 'bg-emerald-400 animate-ping' : 'bg-slate-400 animate-pulse'}`}
                />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-current" />
              </span>
              {modelBadge.label}
            </span>
          </div>
        </header>

        <div className="grid gap-6 md:grid-cols-[minmax(0,1fr)_380px] md:items-start">
          <div className="flex flex-col gap-6">
            <div className="glass-panel rounded-3xl border border-white/60 p-6">
              <p className="text-sm uppercase tracking-[0.3em] text-slate-400">Choose input</p>
              <h2 className="mt-2 text-2xl font-semibold text-slate-900">How would you like to add a photo?</h2>
              <p className="mt-2 text-sm text-slate-500">
                Capture a new image with your camera or upload one from your library. We&apos;ll resize and
                normalize it automatically for MobileNetV3.
              </p>
              <div className="mt-6 grid gap-4 sm:grid-cols-2">
                <button
                  onClick={() => cameraInputRef.current?.click()}
                  className="group rounded-2xl border border-[#4B9CD3]/30 bg-white/80 p-4 text-left transition hover:-translate-y-0.5 hover:border-[#4B9CD3] hover:bg-white"
                >
                  <div className="inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-[#4B9CD3]/15 text-[#4B9CD3]">
                    {cameraIcon}
                  </div>
                  <p className="mt-4 text-lg font-semibold text-slate-900">Take photo</p>
                  <p className="text-sm text-slate-500">Use your device camera with one tap.</p>
                </button>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="group rounded-2xl border border-slate-200 bg-white/70 p-4 text-left transition hover:-translate-y-0.5 hover:border-[#13294B]/70 hover:bg-white"
                >
                  <div className="inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-[#13294B]/10 text-[#13294B]">
                    {uploadIcon}
                  </div>
                  <p className="mt-4 text-lg font-semibold text-slate-900">Upload from library</p>
                  <p className="text-sm text-slate-500">Pick an existing photo of campus.</p>
                </button>
              </div>
              <input
                ref={cameraInputRef}
                type="file"
                accept="image/*"
                capture="environment"
                className="hidden"
                onChange={handleFileChange}
              />
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleFileChange}
              />
            </div>

            <div className="relative overflow-hidden rounded-3xl border border-dashed border-slate-200 bg-white/70 shadow-inner">
              {previewUrl ? (
                <img
                  src={previewUrl}
                  alt="Selected landmark preview"
                  className="h-[380px] w-full object-cover"
                />
              ) : (
                <div className="flex h-[380px] flex-col items-center justify-center gap-4 text-center text-slate-400">
                  <div className="h-20 w-20 rounded-full border border-dashed border-slate-300/80 p-6">
                    <div className="h-full w-full rounded-full bg-gradient-to-br from-[#4B9CD3]/20 to-[#13294B]/20" />
                  </div>
                  <p className="text-lg font-medium text-slate-500">Your preview will show up here.</p>
                  <p className="max-w-sm text-sm text-slate-400">
                    We&apos;ll keep your image on-device and run the AI model locally.
                  </p>
                </div>
              )}

              {analysisState === 'processing' && (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-slate-950/60 text-white">
                  <div className="h-1 w-1/2 overflow-hidden rounded-full bg-white/15">
                    <div className="h-full w-full animate-pulse bg-gradient-to-r from-[#4B9CD3] via-white to-[#13294B]" />
                  </div>
                  <p className="text-sm font-medium uppercase tracking-[0.4em]">Analyzing landmark...</p>
                </div>
              )}
            </div>
          </div>

          <aside className="fixed inset-x-0 bottom-0 z-30 glass-panel rounded-t-3xl border border-white/40 bg-white/95 p-6 shadow-[0_-20px_50px_rgba(15,23,42,0.25)] md:static md:rounded-3xl md:shadow-[0_20px_50px_rgba(15,23,42,0.15)]">
            <p className="text-sm uppercase tracking-[0.3em] text-slate-400">Result</p>
            <div className="mt-4 flex items-center justify-between">
              <h3 className="text-2xl font-semibold text-slate-900">
                {prediction ? prediction.name : 'Awaiting photo'}
              </h3>
              {prediction && (
                <span className="rounded-full bg-[#4B9CD3]/15 px-3 py-1 text-xs font-semibold text-[#13294B]">
                  {confidencePercent}% match
                </span>
              )}
            </div>
            <p className="mt-2 text-sm text-slate-500">{statusMessage}</p>

            <div className="mt-6 space-y-4 text-sm text-slate-600">
              <p>{prediction ? prediction.description : 'Snap a landmark and we will surface its story here.'}</p>
              <div className="rounded-2xl bg-slate-100/80 p-4">
                <p className="text-xs uppercase tracking-[0.4em] text-slate-500">Fun fact</p>
                <p className="mt-2 text-slate-700">
                  {prediction ? prediction.funFact : 'UNC landmarks each hide a tradition. Submit a photo to learn one.'}
                </p>
              </div>
            </div>

            <div className="mt-6 rounded-2xl border border-dashed border-slate-200 p-4 text-xs text-slate-500">
              <p className="font-semibold uppercase tracking-[0.3em] text-slate-400">Model notes</p>
              <ul className="mt-3 list-disc space-y-1 pl-5">
                <li>Powered by MobileNetV3, running fully in your browser.</li>
                <li>Images never leave your device—perfect for on-the-go explorers.</li>
                <li>Try different angles or lighting if the match looks off.</li>
              </ul>
            </div>
          </aside>
        </div>
      </div>
    </section>
  )
}

export default Inference

