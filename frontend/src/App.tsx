/**
 * Top-level screen: holds the two-eye visit, talks to the API and wires the three panels together.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from './api'
import { PrintReport } from './components/PrintReport'
import { ReportPanel } from './components/ReportPanel'
import { ShortcutsDialog } from './components/ShortcutsDialog'
import { TopBar } from './components/TopBar'
import { VisitPanel } from './components/VisitPanel'
import { Viewer } from './components/Viewer'
import type { ClinicianReview, EyeCase, EyeSide, HealthResponse, Sample, ViewMode } from './types'

type Theme = 'light' | 'dark'
const THEME_KEY = 'retinascan-theme'
const EMPTY_VISIT: Record<EyeSide, EyeCase | null> = { right: null, left: null }

/** Recover stored theme or fall back to system preference. */
function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem(THEME_KEY)
    if (saved === 'light' || saved === 'dark') return saved
  } catch {
    // storage can be unavailable in private windows
  }
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

/** Parse view mode from URL query parameter, default to original. */
function initialView(): ViewMode {
  const requested = new URLSearchParams(window.location.search).get('view')
  return requested === 'enhanced' || requested === 'compare' || requested === 'heatmap'
    ? requested
    : 'original'
}

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [serverError, setServerError] = useState<string | null>(null)
  const [samples, setSamples] = useState<Sample[]>([])
  const [patientRef, setPatientRef] = useState('')
  const [eyes, setEyes] = useState<Record<EyeSide, EyeCase | null>>(EMPTY_VISIT)
  const [active, setActive] = useState<EyeSide>('right')
  const [view, setView] = useState<ViewMode>(initialView)
  const [opacity, setOpacity] = useState(0.55)
  const [loupe, setLoupe] = useState(false)
  const [theme, setTheme] = useState<Theme>(initialTheme)
  const [shortcutsOpen, setShortcutsOpen] = useState(false)
  const tokens = useRef<Record<EyeSide, number>>({ right: 0, left: 0 })

  useEffect(() => {
    api.health().then(setHealth).catch((e: Error) => setServerError(e.message))
    api.samples().then(setSamples).catch(() => setSamples([]))
  }, [])


  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try {
      localStorage.setItem(THEME_KEY, theme)
    } catch {
      // ignore unavailable storage
    }
  }, [theme])

  /** Merge updates into an eye's result; no-op if the eye slot is empty. */
  const patch = useCallback((side: EyeSide, update: Partial<EyeCase>) => {
    setEyes((prev) => (prev[side] ? { ...prev, [side]: { ...prev[side]!, ...update } } : prev))
  }, [])

  /**
   * Show the photo straight away, then fill in the result when the API answers. Each eye keeps a
   * counter so a slow response for an old photo cannot overwrite a newer one.
   */
  const analyse = useCallback(
    async (side: EyeSide, blob: Blob, filename: string, sample: Sample | null) => {
      const token = ++tokens.current[side]
      setEyes((prev) => {
        if (prev[side]) URL.revokeObjectURL(prev[side]!.preview)
        return {
          ...prev,
          [side]: {
            side,
            filename,
            preview: URL.createObjectURL(blob),
            sample,
            status: 'analysing',
            result: null,
            error: null,
            review: null,
          },
        }
      })
      setActive(side)
      try {
        const result = await api.predict(blob, filename)
        if (token === tokens.current[side]) patch(side, { status: 'done', result })
      } catch (e) {
        if (token === tokens.current[side]) patch(side, { status: 'error', error: (e as Error).message })
      }
    },
    [patch],
  )

  /** ?right=id&left=id&view=mode loads sample photos on page load for demos. */
  const preloaded = useRef(false)
  useEffect(() => {
    if (preloaded.current || samples.length === 0) return
    preloaded.current = true
    const params = new URLSearchParams(window.location.search)
    for (const side of ['left', 'right'] as EyeSide[]) {
      const sample = samples.find((s) => s.id === params.get(side))
      if (sample) {
        api
          .sampleImage(sample.id)
          .then((blob) => analyse(side, blob, `${sample.id}.jpg`, sample))
          .catch(() => undefined)
      }
    }
  }, [samples, analyse])

  const onSample = async (sample: Sample) => {
    const side = active
    try {
      const blob = await api.sampleImage(sample.id)
      await analyse(side, blob, `${sample.id}.jpg`, sample)
    } catch (e) {
      setServerError((e as Error).message)
    }
  }

  /** Revoke the preview blob and reset the eye slot. */
  const clear = (side: EyeSide) => {
    tokens.current[side]++
    setEyes((prev) => {
      if (prev[side]) URL.revokeObjectURL(prev[side]!.preview)
      return { ...prev, [side]: null }
    })
  }

  /** Clear both eyes, reset patient reference and view to defaults. */
  const newVisit = () => {
    clear('right')
    clear('left')
    setPatientRef('')
    setActive('right')
    setView('original')
  }

  /** Top-level keyboard shortcuts: ? for help, R/E to switch eyes. */
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement
      if (target.closest('input, textarea, select, [contenteditable]') || event.metaKey || event.ctrlKey)
        return
      if (event.key === '?') setShortcutsOpen(true)
      else if (event.key.toLowerCase() === 'r') setActive('right')
      else if (event.key.toLowerCase() === 'e') setActive('left')
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  const anyDone = Boolean(eyes.right?.result || eyes.left?.result)
  const busy = eyes.right?.status === 'analysing' || eyes.left?.status === 'analysing'

  return (
    <>
      <div className="app">
        <TopBar
          health={health}
          serverError={serverError}
          theme={theme}
          onToggleTheme={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
          onShortcuts={() => setShortcutsOpen(true)}
          onPrint={() => window.print()}
          canPrint={anyDone}
          onNewVisit={newVisit}
        />
        {serverError && (
          <p className="banner banner-fail" role="alert">
            {serverError}
          </p>
        )}
        {health?.demo && (
          <p className="banner banner-warn">
            No trained model is loaded, so grades are simulated. Start the server with{' '}
            <code>--checkpoint</code> to use the trained model.
          </p>
        )}
        <main className="workspace">
          <VisitPanel
            patientRef={patientRef}
            onPatientRefChange={setPatientRef}
            eyes={eyes}
            active={active}
            onSelect={setActive}
            onFile={(side, file) => void analyse(side, file, file.name, null)}
            onClear={clear}
            samples={samples}
            onSample={onSample}
            busy={Boolean(busy)}
          />
          <Viewer
            key={eyes[active]?.preview ?? active}
            eye={eyes[active]}
            side={active}
            view={view}
            onViewChange={setView}
            opacity={opacity}
            onOpacityChange={setOpacity}
            loupe={loupe}
            onLoupeChange={setLoupe}
            onFile={(file) => void analyse(active, file, file.name, null)}
          />
          <ReportPanel
            eyes={eyes}
            active={active}
            onSelect={setActive}
            onReview={(side, review: ClinicianReview | null) => patch(side, { review })}
            health={health}
          />
        </main>
      </div>
      <PrintReport patientRef={patientRef} eyes={eyes} health={health} />
      <ShortcutsDialog open={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />
    </>
  )
}
