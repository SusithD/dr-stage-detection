import { useCallback, useEffect, useRef, useState } from 'react'
import type { KeyboardEvent as ReactKeyboardEvent, PointerEvent, WheelEvent } from 'react'
import type { EyeCase, EyeSide, ViewMode } from '../types'
import { FitIcon, LoupeIcon, UploadIcon } from './Icons'

const VIEWS: { value: ViewMode; label: string; key: string }[] = [
  { value: 'original', label: 'Original', key: '1' },
  { value: 'enhanced', label: 'Enhanced', key: '2' },
  { value: 'compare', label: 'Compare', key: '3' },
  { value: 'heatmap', label: 'Heatmap', key: '4' },
]
const MIN_ZOOM = 1
const MAX_ZOOM = 6
const LOUPE_SIZE = 180
const LOUPE_ZOOM = 3

interface ViewerProps {
  eye: EyeCase | null
  side: EyeSide
  view: ViewMode
  onViewChange: (view: ViewMode) => void
  opacity: number
  onOpacityChange: (value: number) => void
  loupe: boolean
  onLoupeChange: (on: boolean) => void
  onFile: (file: File) => void
}

export function Viewer({
  eye,
  side,
  view,
  onViewChange,
  opacity,
  onOpacityChange,
  loupe,
  onLoupeChange,
  onFile,
}: ViewerProps) {
  const stage = useRef<HTMLDivElement>(null)
  const drag = useRef<{ x: number; y: number; px: number; py: number } | null>(null)
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [split, setSplit] = useState(50)
  const [pointer, setPointer] = useState<{ x: number; y: number } | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const [stageSize, setStageSize] = useState<{ width: number; height: number } | null>(null)

  useEffect(() => {
    const el = stage.current
    if (!el) return
    const observer = new ResizeObserver(([entry]) =>
      setStageSize({ width: entry.contentRect.width, height: entry.contentRect.height }),
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  const images = eye?.result?.images ?? null
  const base = images ? (view === 'enhanced' ? images.enhanced : images.original) : eye?.preview ?? null
  const ready = Boolean(images)

  const reset = useCallback(() => {
    setZoom(1)
    setPan({ x: 0, y: 0 })
  }, [])

  const zoomBy = useCallback((factor: number) => {
    setZoom((z) => {
      const next = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, z * factor))
      if (next === 1) setPan({ x: 0, y: 0 })
      return next
    })
  }, [])

  const onWheel = (event: WheelEvent) => {
    if (!ready) return
    zoomBy(event.deltaY < 0 ? 1.15 : 1 / 1.15)
  }

  const onPointerDown = (event: PointerEvent) => {
    if (!ready || zoom === 1 || (event.target as HTMLElement).closest('.split-handle')) return
    event.currentTarget.setPointerCapture(event.pointerId)
    drag.current = { x: event.clientX, y: event.clientY, px: pan.x, py: pan.y }
  }

  const onPointerMove = (event: PointerEvent) => {
    const rect = stage.current?.getBoundingClientRect()
    if (rect) setPointer({ x: event.clientX - rect.left, y: event.clientY - rect.top })
    if (drag.current) {
      setPan({
        x: drag.current.px + (event.clientX - drag.current.x) / zoom,
        y: drag.current.py + (event.clientY - drag.current.y) / zoom,
      })
    }
  }

  const endDrag = () => {
    drag.current = null
  }

  const onKeyDown = (event: ReactKeyboardEvent) => {
    if (view !== 'compare') return
    if (event.key === 'ArrowLeft') setSplit((s) => Math.max(0, s - 5))
    if (event.key === 'ArrowRight') setSplit((s) => Math.min(100, s + 5))
  }

  const startSplitDrag = (event: PointerEvent) => {
    event.stopPropagation()
    const move = (e: globalThis.PointerEvent) => {
      const rect = stage.current?.getBoundingClientRect()
      if (!rect) return
      setSplit(Math.min(100, Math.max(0, ((e.clientX - rect.left) / rect.width) * 100)))
    }
    const up = () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
    }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
  }

  useEffect(() => {
    const handler = (event: globalThis.KeyboardEvent) => {
      const target = event.target as HTMLElement
      if (target.closest('input, textarea, select, [contenteditable]') || event.metaKey || event.ctrlKey)
        return
      const match = VIEWS.find((v) => v.key === event.key)
      if (match && ready) onViewChange(match.value)
      else if ((event.key === '+' || event.key === '=') && ready) zoomBy(1.25)
      else if (event.key === '-' && ready) zoomBy(0.8)
      else if (event.key === '0') reset()
      else if (event.key.toLowerCase() === 'l' && ready) onLoupeChange(!loupe)
      else if (event.key === '[') onOpacityChange(Math.max(0, +(opacity - 0.1).toFixed(2)))
      else if (event.key === ']') onOpacityChange(Math.min(1, +(opacity + 0.1).toFixed(2)))
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [ready, loupe, opacity, onViewChange, onLoupeChange, onOpacityChange, zoomBy, reset])

  const transform = `scale(${zoom}) translate(${pan.x}px, ${pan.y}px)`
  const showLoupe = loupe && ready && pointer && stageSize && view !== 'compare'

  return (
    <section className="viewer" aria-label="Photograph viewer">
      <div className="viewer-toolbar" role="toolbar" aria-label="Viewer tools">
        <div className="seg" role="radiogroup" aria-label="Image view">
          {VIEWS.map((v) => (
            <button
              key={v.value}
              type="button"
              role="radio"
              aria-checked={view === v.value}
              className="seg-btn"
              disabled={!ready}
              onClick={() => onViewChange(v.value)}
              title={`${v.label} (${v.key})`}
            >
              {v.label}
            </button>
          ))}
        </div>

        <label className="tool-range" data-disabled={!ready || view !== 'heatmap' || undefined}>
          <span>Heatmap</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={opacity}
            disabled={!ready || view !== 'heatmap'}
            onChange={(e) => onOpacityChange(Number(e.target.value))}
            aria-label="Heatmap opacity"
          />
          <output>{Math.round(opacity * 100)}%</output>
        </label>

        <div className="tool-group">
          <button
            type="button"
            className="icon-btn"
            aria-pressed={loupe}
            disabled={!ready || view === 'compare'}
            onClick={() => onLoupeChange(!loupe)}
            title="Magnifier (L)"
          >
            <LoupeIcon />
            <span className="visually-hidden">Magnifier</span>
          </button>
          <button type="button" className="icon-btn" disabled={!ready} onClick={() => zoomBy(0.8)} title="Zoom out (−)">
            <span aria-hidden="true">−</span>
            <span className="visually-hidden">Zoom out</span>
          </button>
          <output className="zoom-readout" aria-label="Zoom level">
            {Math.round(zoom * 100)}%
          </output>
          <button type="button" className="icon-btn" disabled={!ready} onClick={() => zoomBy(1.25)} title="Zoom in (+)">
            <span aria-hidden="true">+</span>
            <span className="visually-hidden">Zoom in</span>
          </button>
          <button type="button" className="icon-btn" disabled={!ready || zoom === 1} onClick={reset} title="Fit (0)">
            <FitIcon />
            <span className="visually-hidden">Fit to view</span>
          </button>
        </div>
      </div>

      <div className="stage-wrap">
      <div
        ref={stage}
        className="stage"
        data-zoomed={zoom > 1 || undefined}
        data-dragover={dragOver || undefined}
        data-loupe={showLoupe || undefined}
        tabIndex={ready ? 0 : -1}
        aria-label={ready ? 'Fundus photograph. Scroll to zoom, drag to pan.' : undefined}
        onWheel={onWheel}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onPointerLeave={() => setPointer(null)}
        onDoubleClick={() => (zoom > 1 ? reset() : zoomBy(2))}
        onKeyDown={onKeyDown}
        onDragOver={(e) => {
          e.preventDefault()
          setDragOver(true)
        }}
        onDragLeave={(e) => {
          if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setDragOver(false)
        }}
        onDrop={(e) => {
          e.preventDefault()
          setDragOver(false)
          const file = e.dataTransfer.files[0]
          if (file) onFile(file)
        }}
      >
        {base ? (
          <div className="canvas" style={{ transform }}>
            {view === 'compare' && images ? (
              <>
                <img className="layer" src={images.original} alt="Original photograph" draggable={false} />
                <img
                  className="layer"
                  src={images.enhanced}
                  alt="Enhanced photograph"
                  draggable={false}
                  style={{ clipPath: `inset(0 0 0 ${split}%)` }}
                />
              </>
            ) : (
              <>
                <img
                  className="layer"
                  src={base}
                  alt={view === 'enhanced' ? 'Enhanced photograph as seen by the model' : 'Fundus photograph'}
                  draggable={false}
                  data-dim={eye?.status === 'analysing' || undefined}
                />
                {images && view === 'heatmap' && (
                  <img
                    className="layer heat"
                    src={images.heatmap}
                    alt="Regions that most influenced the grade"
                    draggable={false}
                    style={{ opacity }}
                  />
                )}
              </>
            )}
          </div>
        ) : (
          <div className="stage-empty">
            <UploadIcon className="stage-empty-icon" />
            <p>
              Drop a fundus photograph of the <strong>{side} eye</strong> here,
              or choose one from the visit panel.
            </p>
          </div>
        )}

        {view === 'compare' && images && (
          <div className="split" style={{ left: `${split}%` }}>
            <button
              type="button"
              className="split-handle"
              onPointerDown={startSplitDrag}
              aria-label={`Comparison divider at ${Math.round(split)}%. Use arrow keys to move.`}
            />
            <span className="split-tag split-tag-left">Original</span>
            <span className="split-tag split-tag-right">Enhanced</span>
          </div>
        )}

        {showLoupe && (
          <div
            className="loupe"
            aria-hidden="true"
            style={{
              width: LOUPE_SIZE,
              height: LOUPE_SIZE,
              left: pointer.x - LOUPE_SIZE / 2,
              top: pointer.y - LOUPE_SIZE / 2,
              backgroundImage:
                view === 'heatmap' && images
                  ? `url(${images.heatmap}), url(${images.original})`
                  : `url(${base})`,
              backgroundSize: `${stageSize.width * zoom * LOUPE_ZOOM}px ${stageSize.height * zoom * LOUPE_ZOOM}px`,
              backgroundPosition: loupePosition(pointer, stageSize, zoom, pan),
            }}
          />
        )}

        {eye?.status === 'analysing' && (
          <div className="scanline" role="progressbar" aria-label="Analysing photograph" />
        )}
      </div>

      </div>

      <p className="viewer-hint">
        {ready
          ? 'Scroll to zoom, drag to pan, double-click to magnify. Press ? for all shortcuts.'
          : 'Photographs are analysed on this computer and are not stored.'}
      </p>
    </section>
  )
}

function loupePosition(
  pointer: { x: number; y: number },
  size: { width: number; height: number },
  zoom: number,
  pan: { x: number; y: number },
): string {
  const scale = zoom * LOUPE_ZOOM
  const cx = size.width / 2
  const cy = size.height / 2
  // map the pointer back to image coordinates, undoing the canvas zoom and pan
  const ix = (pointer.x - cx) / zoom - pan.x + cx
  const iy = (pointer.y - cy) / zoom - pan.y + cy
  const x = -(ix * scale - LOUPE_SIZE / 2)
  const y = -(iy * scale - LOUPE_SIZE / 2)
  return `${x}px ${y}px`
}
