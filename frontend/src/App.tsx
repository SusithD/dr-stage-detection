import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from './api'
import { DemoBanner } from './components/DemoBanner'
import { Findings } from './components/Findings'
import { SampleStrip } from './components/SampleStrip'
import { ViewToggle } from './components/ViewToggle'
import { Viewport } from './components/Viewport'
import { asGrade, GRADES } from './grades'
import type { HealthResponse, PredictResponse, Sample, ViewMode } from './types'

type Status = 'idle' | 'loading' | 'done' | 'error'

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [serverError, setServerError] = useState<string | null>(null)
  const [samples, setSamples] = useState<Sample[]>([])
  const [status, setStatus] = useState<Status>('idle')
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [activeSample, setActiveSample] = useState<Sample | null>(null)
  const [view, setView] = useState<ViewMode>('original')
  const [opacity, setOpacity] = useState(0.6)
  const fileInput = useRef<HTMLInputElement>(null)
  const requestId = useRef(0)

  useEffect(() => {
    api
      .health()
      .then(setHealth)
      .catch((e: Error) => setServerError(e.message))
    api
      .samples()
      .then(setSamples)
      .catch(() => setSamples([]))
  }, [])

  useEffect(() => () => {
    if (preview) URL.revokeObjectURL(preview)
  }, [preview])

  const analyse = useCallback(async (blob: Blob, filename: string, sample: Sample | null) => {
    const id = ++requestId.current
    setPreview(URL.createObjectURL(blob))
    setActiveSample(sample)
    setResult(null)
    setError(null)
    setStatus('loading')
    try {
      const response = await api.predict(blob, filename)
      if (id !== requestId.current) return
      setResult(response)
      setStatus('done')
    } catch (e) {
      if (id !== requestId.current) return
      setError((e as Error).message)
      setStatus('error')
    }
  }, [])

  const onFile = (file: File) => void analyse(file, file.name, null)

  const onSample = async (sample: Sample) => {
    try {
      const blob = await api.sampleImage(sample.id)
      await analyse(blob, `${sample.id}.jpg`, sample)
    } catch (e) {
      setError((e as Error).message)
      setStatus('error')
    }
  }

  const reset = () => {
    requestId.current++
    setStatus('idle')
    setResult(null)
    setError(null)
    setPreview(null)
    setActiveSample(null)
  }

  return (
    <div className="app">
      <header className="masthead">
        <h1 className="brand">RetinaScan</h1>
        <p className="tagline">Diabetic retinopathy screening from fundus photographs</p>
      </header>

      {health?.demo && <DemoBanner />}
      {serverError && (
        <p className="server-error" role="alert">
          {serverError}
        </p>
      )}

      <main className="workspace">
        <section className="examine" aria-label="Photograph">
          <Viewport
            preview={preview}
            images={result?.images ?? null}
            view={view}
            heatmapOpacity={opacity}
            loading={status === 'loading'}
            onFile={onFile}
          />
          <div className="examine-controls">
            <ViewToggle
              view={view}
              onViewChange={setView}
              heatmapOpacity={opacity}
              onOpacityChange={setOpacity}
              disabled={!result}
            />
            <button
              type="button"
              className="button-primary"
              onClick={() => fileInput.current?.click()}
              disabled={status === 'loading' || !!serverError}
            >
              Upload a photograph
            </button>
            <input
              ref={fileInput}
              type="file"
              accept=".jpg,.jpeg,.png,.tif,.tiff"
              className="visually-hidden"
              tabIndex={-1}
              onChange={(event) => {
                const file = event.target.files?.[0]
                if (file) onFile(file)
                event.target.value = ''
              }}
            />
          </div>
        </section>

        <section className="report" aria-live="polite" aria-busy={status === 'loading'}>
          {status === 'idle' && (
            <p className="report-empty">
              Upload a fundus photograph or choose a sample to see findings.
            </p>
          )}
          {status === 'loading' && <p className="report-empty">Analysing the photograph…</p>}
          {status === 'error' && (
            <div className="report-error">
              <h2>The photograph couldn't be analysed</h2>
              <p>{error}</p>
              <button type="button" className="button-secondary" onClick={reset}>
                Try another image
              </button>
            </div>
          )}
          {status === 'done' && result && (
            <>
              <Findings result={result} health={health} />
              {activeSample && <GroundTruth sample={activeSample} predicted={result.grade} />}
            </>
          )}
        </section>
      </main>

      <SampleStrip
        samples={samples}
        activeId={activeSample?.id ?? null}
        disabled={status === 'loading'}
        onSelect={onSample}
      />

      <footer className="colophon">
        Research prototype for a university coursework project. Not a medical device and not for
        clinical use.
      </footer>
    </div>
  )
}

function GroundTruth({ sample, predicted }: { sample: Sample; predicted: number }) {
  const match = sample.label === predicted
  return (
    <p className="ground-truth" data-match={match}>
      <svg viewBox="0 0 20 20" aria-hidden="true">
        {match ? <path d="M4 10.5l4 4 8-9" /> : <path d="M5 5l10 10M15 5L5 15" />}
      </svg>
      Ground truth grade: {GRADES[asGrade(sample.label)].short}. Model prediction:{' '}
      {GRADES[asGrade(predicted)].short}.
    </p>
  )
}
