import { useEffect, useState } from 'react'
import { GRADE_LIST, GRADES } from '../grades'
import type { Grade } from '../types'

interface SeverityScaleProps {
  grade: Grade
  classProbs: number[]
}

export function SeverityScale({ grade, classProbs }: SeverityScaleProps) {
  const [markerGrade, setMarkerGrade] = useState<number>(0)

  useEffect(() => {
    const frame = requestAnimationFrame(() => setMarkerGrade(grade))
    return () => cancelAnimationFrame(frame)
  }, [grade])

  return (
    <figure className="scale">
      <figcaption className="scale-caption">International clinical severity scale</figcaption>
      <div className="scale-track" role="list">
        {GRADE_LIST.map((g) => {
          const info = GRADES[g]
          const prob = classProbs[g] ?? 0
          const active = g === grade
          return (
            <div
              key={g}
              role="listitem"
              className="scale-step"
              data-active={active || undefined}
              aria-label={`Grade ${g}, ${info.short}: ${Math.round(prob * 100)}%`}
            >
              <div className="scale-swatch" style={{ background: info.color, color: info.onColor }}>
                {g}
              </div>
              <div className="scale-name">{info.short}</div>
              <div className="scale-bar" aria-hidden="true">
                <div className="scale-bar-fill" style={{ height: `${prob * 100}%` }} />
              </div>
              <div className="scale-prob">{Math.round(prob * 100)}%</div>
            </div>
          )
        })}
        <div
          className="scale-marker"
          aria-hidden="true"
          style={{ left: `${((markerGrade + 0.5) / GRADE_LIST.length) * 100}%` }}
        />
      </div>
    </figure>
  )
}
