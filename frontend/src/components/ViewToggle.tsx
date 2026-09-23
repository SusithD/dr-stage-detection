import type { ViewMode } from '../types'

const VIEWS: { value: ViewMode; label: string }[] = [
  { value: 'original', label: 'Original' },
  { value: 'enhanced', label: 'Enhanced' },
  { value: 'heatmap', label: 'Heatmap' },
]

interface ViewToggleProps {
  view: ViewMode
  onViewChange: (view: ViewMode) => void
  heatmapOpacity: number
  onOpacityChange: (opacity: number) => void
  disabled: boolean
}

export function ViewToggle({
  view,
  onViewChange,
  heatmapOpacity,
  onOpacityChange,
  disabled,
}: ViewToggleProps) {
  return (
    <div className="view-controls">
      <fieldset className="segmented" disabled={disabled}>
        <legend className="visually-hidden">Image view</legend>
        {VIEWS.map(({ value, label }) => (
          <label key={value} className="segmented-option">
            <input
              type="radio"
              name="view"
              value={value}
              checked={view === value}
              onChange={() => onViewChange(value)}
            />
            <span>{label}</span>
          </label>
        ))}
      </fieldset>
      <label className="opacity-control" data-disabled={disabled || view !== 'heatmap' || undefined}>
        <span>Heatmap opacity</span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={heatmapOpacity}
          disabled={disabled || view !== 'heatmap'}
          onChange={(event) => onOpacityChange(Number(event.target.value))}
        />
      </label>
    </div>
  )
}
