/**
 * Left sidebar: patient ID field, eye photo slots with upload buttons, and test-set sample gallery.
 * Sample grid is sorted by grade so each column shows one severity level.
 */
import { useRef } from 'react'
import { asGrade, finalGrade } from '../grades'
import type { EyeCase, EyeSide, Sample } from '../types'
import { CloseIcon, StatusIcon, UploadIcon } from './Icons'
import { SeverityPips } from './SeverityPips'

const SIDES: EyeSide[] = ['right', 'left']

interface VisitPanelProps {
  patientRef: string
  onPatientRefChange: (value: string) => void
  eyes: Record<EyeSide, EyeCase | null>
  active: EyeSide
  onSelect: (side: EyeSide) => void
  onFile: (side: EyeSide, file: File) => void
  onClear: (side: EyeSide) => void
  samples: Sample[]
  onSample: (sample: Sample) => void
  busy: boolean
}

export function VisitPanel({
  patientRef,
  onPatientRefChange,
  eyes,
  active,
  onSelect,
  onFile,
  onClear,
  samples,
  onSample,
  busy,
}: VisitPanelProps) {
  return (
    <aside className="visit" aria-label="Screening visit">
      <div className="visit-block">
        <label className="field">
          <span className="field-label">Patient reference</span>
          <input
            className="input"
            value={patientRef}
            onChange={(e) => onPatientRefChange(e.target.value)}
            placeholder="Optional, e.g. clinic number"
            maxLength={40}
            autoComplete="off"
          />
        </label>
      </div>

      <div className="visit-block">
        <h2 className="panel-heading">Photographs</h2>
        <ul className="eye-list">
          {SIDES.map((side) => (
            <EyeSlot
              key={side}
              side={side}
              eye={eyes[side]}
              selected={active === side}
              onSelect={() => onSelect(side)}
              onFile={(file) => onFile(side, file)}
              onClear={() => onClear(side)}
            />
          ))}
        </ul>
      </div>

      {samples.length > 0 && (
        <div className="visit-block visit-samples">
          <h2 className="panel-heading">Test-set photographs</h2>
          <p className="panel-note">
            Held out from training. Choosing one loads it into the <strong>{active} eye</strong>{' '}
            slot; its reference grade is shown for comparison.
          </p>
          <ul className="sample-grid">
            {byGradeColumns(samples).map((s) => (
              <li key={s.id}>
                <button
                  type="button"
                  className="sample"
                  disabled={busy}
                  onClick={() => onSample(s)}
                  aria-label={`Load test photograph with reference grade ${s.grade_name} into the ${active} eye`}
                >
                  <img src={s.thumbnail} alt="" loading="lazy" className="sample-img" />
                  <SeverityPips grade={asGrade(s.label)} size="sm" />
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </aside>
  )
}

interface EyeSlotProps {
  side: EyeSide
  eye: EyeCase | null
  selected: boolean
  onSelect: () => void
  onFile: (file: File) => void
  onClear: () => void
}

function EyeSlot({ side, eye, selected, onSelect, onFile, onClear }: EyeSlotProps) {
  const input = useRef<HTMLInputElement>(null)
  const label = side === 'right' ? 'Right eye' : 'Left eye'
  const grade = eye ? finalGrade(eye) : null

  return (
    <li className="eye" data-selected={selected || undefined}>
      <button type="button" className="eye-main" onClick={onSelect} aria-pressed={selected}>
        <span className="eye-thumb" data-empty={!eye || undefined}>
          {eye ? <img src={eye.result?.images.original ?? eye.preview} alt="" /> : <span aria-hidden="true">{side === 'right' ? 'R' : 'L'}</span>}
        </span>
        <span className="eye-text">
          <span className="eye-name">
            {label}
            <span className="eye-abbr">{side === 'right' ? 'OD' : 'OS'}</span>
          </span>
          <span className="eye-status">
            {!eye && 'No photograph'}
            {eye?.status === 'analysing' && 'Analysing…'}
            {eye?.status === 'error' && <span className="text-fail">Couldn't be analysed</span>}
            {eye?.status === 'done' && grade !== null && (
              <>
                <SeverityPips grade={grade} size="sm" showLabel />
                {eye.result && !eye.result.quality.gradable && !eye.review && (
                  <span className="eye-flag">
                    <StatusIcon status="fail" /> Quality
                  </span>
                )}
                {eye.review && <span className="eye-flag eye-flag-review">Reviewed</span>}
              </>
            )}
          </span>
        </span>
      </button>
      <div className="eye-actions">
        <button
          type="button"
          className="icon-btn"
          onClick={() => input.current?.click()}
          title={`Upload ${label.toLowerCase()} photograph`}
        >
          <UploadIcon />
          <span className="visually-hidden">Upload {label.toLowerCase()} photograph</span>
        </button>
        {eye && (
          <button type="button" className="icon-btn" onClick={onClear} title={`Remove ${label.toLowerCase()} photograph`}>
            <CloseIcon />
            <span className="visually-hidden">Remove {label.toLowerCase()} photograph</span>
          </button>
        )}
      </div>
      <input
        ref={input}
        type="file"
        accept=".jpg,.jpeg,.png,.tif,.tiff"
        className="visually-hidden"
        tabIndex={-1}
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) onFile(file)
          e.target.value = ''
        }}
      />
    </li>
  )
}

/** Put the first sample of each grade in row one, the second in row two, so columns line up by grade. */
function byGradeColumns(samples: Sample[]): Sample[] {
  const seen = new Map<number, number>()
  return samples
    .map((s) => {
      const rank = seen.get(s.label) ?? 0
      seen.set(s.label, rank + 1)
      return { s, rank }
    })
    .sort((a, b) => a.rank - b.rank || a.s.label - b.s.label)
    .map(({ s }) => s)
}
