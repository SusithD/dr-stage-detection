import { asGrade, GRADES } from '../grades'
import type { Sample } from '../types'

interface SampleStripProps {
  samples: Sample[]
  activeId: string | null
  disabled: boolean
  onSelect: (sample: Sample) => void
}

export function SampleStrip({ samples, activeId, disabled, onSelect }: SampleStripProps) {
  if (samples.length === 0) return null

  return (
    <section className="samples" aria-labelledby="samples-heading">
      <h2 id="samples-heading" className="samples-heading">
        Try a photograph from the held-out test set
      </h2>
      <ul className="samples-list">
        {samples.map((sample) => {
          const info = GRADES[asGrade(sample.label)]
          return (
            <li key={sample.id}>
              <button
                type="button"
                className="sample"
                aria-pressed={sample.id === activeId}
                disabled={disabled}
                onClick={() => onSelect(sample)}
                title={`Ground truth grade: ${info.short}`}
              >
                <img src={sample.thumbnail} alt="" className="sample-thumb" loading="lazy" />
                <span className="sample-label">
                  <span className="sample-dot" style={{ background: info.color }} aria-hidden="true" />
                  {info.short}
                  <span className="visually-hidden">, ground truth grade {sample.label}</span>
                </span>
              </button>
            </li>
          )
        })}
      </ul>
    </section>
  )
}
