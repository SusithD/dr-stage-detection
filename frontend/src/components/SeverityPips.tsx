import { GRADES } from '../grades'
import type { Grade } from '../types'

interface SeverityPipsProps {
  grade: Grade
  size?: 'sm' | 'md'
  showLabel?: boolean
}

/** Four bars filled up to the grade, so severity reads without relying on colour. */
export function SeverityPips({ grade, size = 'md', showLabel = false }: SeverityPipsProps) {
  const info = GRADES[grade]
  return (
    <span className={`pips pips-${size}`} style={{ ['--pip' as string]: info.color }}>
      <span className="pips-bars" aria-hidden="true">
        {[1, 2, 3, 4].map((level) => (
          <span key={level} className="pip" data-on={level <= grade || undefined} />
        ))}
      </span>
      {showLabel ? (
        <span className="pips-label">
          {info.short} <span className="pips-grade">grade {grade}</span>
        </span>
      ) : (
        <span className="visually-hidden">
          {info.short}, grade {grade}
        </span>
      )}
    </span>
  )
}
