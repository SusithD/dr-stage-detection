import { useState } from 'react'
import type { DragEvent } from 'react'
import type { ViewMode } from '../types'

interface ViewportProps {
  preview: string | null
  images: { original: string; enhanced: string; heatmap: string } | null
  view: ViewMode
  heatmapOpacity: number
  loading: boolean
  onFile: (file: File) => void
}

export function Viewport({ preview, images, view, heatmapOpacity, loading, onFile }: ViewportProps) {
  const [dragging, setDragging] = useState(false)

  const onDragOver = (event: DragEvent) => {
    event.preventDefault()
    setDragging(true)
  }
  const onDragLeave = (event: DragEvent) => {
    if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragging(false)
  }
  const onDrop = (event: DragEvent) => {
    event.preventDefault()
    setDragging(false)
    const file = event.dataTransfer.files[0]
    if (file) onFile(file)
  }

  const base = images ? (view === 'enhanced' ? images.enhanced : images.original) : preview
  const baseAlt =
    view === 'enhanced' && images
      ? 'Enhanced fundus photograph as seen by the model'
      : 'Fundus photograph'

  return (
    <div
      className="viewport-panel"
      data-dragging={dragging || undefined}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      <div className="eyepiece" data-loading={loading || undefined}>
        {base ? (
          <>
            <img className="eyepiece-image" src={base} alt={baseAlt} />
            {images && view === 'heatmap' && (
              <img
                className="eyepiece-heatmap"
                src={images.heatmap}
                alt="Heatmap of the regions that most influenced the grade"
                style={{ opacity: heatmapOpacity }}
              />
            )}
          </>
        ) : (
          <div className="eyepiece-empty">
            <svg viewBox="0 0 48 48" aria-hidden="true" className="eyepiece-icon">
              <circle cx="24" cy="24" r="21" />
              <path d="M24 32V16M17 23l7-7 7 7" />
            </svg>
            <p>Drop a fundus photograph here, or choose a sample below.</p>
          </div>
        )}
        {loading && <div className="eyepiece-progress" role="progressbar" aria-label="Analysing" />}
      </div>
    </div>
  )
}
