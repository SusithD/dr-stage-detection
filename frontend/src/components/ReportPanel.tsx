/**
 * Right sidebar: visit-level screening recommendation (based on worse eye),
 * per-eye grade table, image quality checks, and clinician review UI.
 */
import { useState } from 'react'
import {
  asGrade,
  finalGrade,
  GRADE_LIST,
  GRADES,
  OVERRIDE_REASONS,
  PREPROCESSING_NAMES,
  uncertainty,
  visitOutcome,
} from '../grades'
import type { ClinicianReview, EyeCase, EyeSide, Grade, HealthResponse } from '../types'
import { StatusIcon } from './Icons'
import { SeverityPips } from './SeverityPips'

interface ReportPanelProps {
  eyes: Record<EyeSide, EyeCase | null>
  active: EyeSide
  onSelect: (side: EyeSide) => void
  onReview: (side: EyeSide, review: ClinicianReview | null) => void
  health: HealthResponse | null
}

export function ReportPanel({ eyes, active, onSelect, onReview, health }: ReportPanelProps) {
  const list = [eyes.right, eyes.left].filter((e): e is EyeCase => e !== null)
  const outcome = visitOutcome(list)
  const eye = eyes[active]
  const missing = (['right', 'left'] as EyeSide[]).filter((s) => !eyes[s])

  return (
    <aside className="report" aria-label="Screening report">
      <section className="outcome" data-tone={outcome.tone} aria-live="polite">
        <p className="outcome-kicker">Screening outcome</p>
        <h2 className="outcome-title">{outcome.title}</h2>
        <p className="outcome-detail">{outcome.detail}</p>
        {outcome.tone !== 'pending' && missing.length === 1 && (
          <p className="outcome-note">Based on the {missing[0] === 'right' ? 'left' : 'right'} eye only.</p>
        )}
      </section>

      {list.length > 0 && (
        <section className="report-block">
          <h3 className="panel-heading">Per eye</h3>
          <table className="eye-table">
            <tbody>
              {(['right', 'left'] as EyeSide[]).map((side) => {
                const e = eyes[side]
                const g = e ? finalGrade(e) : null
                return (
                  <tr
                    key={side}
                    data-active={side === active || undefined}
                    onClick={() => e && onSelect(side)}
                  >
                    <th scope="row">{side === 'right' ? 'Right' : 'Left'}</th>
                    <td>
                      {!e && <span className="muted">Not photographed</span>}
                      {e?.status === 'analysing' && <span className="muted">Analysing…</span>}
                      {e?.status === 'error' && <span className="text-fail">Error</span>}
                      {e?.status === 'done' && g !== null && <SeverityPips grade={g} size="sm" showLabel />}
                    </td>
                    <td className="eye-table-src">
                      {e?.review ? 'Clinician' : e?.status === 'done' ? 'Model' : ''}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </section>
      )}

      {eye?.status === 'error' && (
        <section className="report-block callout callout-fail" role="alert">
          <h3>The {active} eye photograph couldn't be analysed</h3>
          <p>{eye.error}</p>
        </section>
      )}

      {eye?.status === 'done' && eye.result && (
        <EyeFindings key={`${active}-${eye.preview}`} eye={eye} onReview={(r) => onReview(active, r)} />
      )}

      <details className="report-block model-details">
        <summary>About the model</summary>
        <dl>
          {health?.model ? (
            <>
              <dt>Network</dt>
              <dd>{health.model.name}</dd>
              <dt>Output</dt>
              <dd>{health.model.head === 'corn' ? 'Ordinal (CORN), grades 0 to 4' : 'Softmax, grades 0 to 4'}</dd>
              <dt>Input</dt>
              <dd>
                {health.model.image_size} × {health.model.image_size} px,{' '}
                {PREPROCESSING_NAMES[health.model.preprocessing] ?? health.model.preprocessing}
              </dd>
            </>
          ) : (
            <>
              <dt>Network</dt>
              <dd>None loaded, demo mode</dd>
            </>
          )}
          <dt>Device</dt>
          <dd>{health?.device ?? 'unknown'}</dd>
          <dt>Training data</dt>
          <dd>DDR, 147 hospitals in China; external test on APTOS 2019 (India)</dd>
          {eye?.result && (
            <>
              <dt>Analysis time</dt>
              <dd>{Math.round(eye.result.inference_ms)} ms</dd>
            </>
          )}
        </dl>
      </details>

      <p className="report-foot">
        Decision support for trained staff. A qualified clinician must confirm every result.
      </p>
    </aside>
  )
}

function EyeFindings({ eye, onReview }: { eye: EyeCase; onReview: (r: ClinicianReview | null) => void }) {
  const result = eye.result!
  const modelGrade = asGrade(result.grade)
  const unsure = uncertainty(result)
  const gradable = result.quality.gradable

  return (
    <>
      <section className="report-block">
        <div className="block-head">
          <h3 className="panel-heading">Image quality</h3>
          <span className="badge" data-status={gradable ? 'ok' : 'fail'}>
            {gradable ? 'Gradable' : 'Ungradable'}
          </span>
        </div>
        <ul className="checks">
          {result.quality.checks.map((c) => (
            <li key={c.key} className="check" data-status={c.status}>
              <StatusIcon status={c.status} />
              <span className="check-label">{c.label}</span>
              <span className="check-msg">{c.message}</span>
            </li>
          ))}
        </ul>
      </section>

      <section className="report-block">
        <div className="block-head">
          <h3 className="panel-heading">Model grade</h3>
          <span className="prob-dr">
            Any retinopathy <strong>{Math.round(result.prob_dr * 100)}%</strong>
          </span>
        </div>
        <Ladder probs={result.class_probs} grade={modelGrade} muted={!gradable} />
        {unsure && (
          <p className="callout callout-warn">
            <StatusIcon status="warn" />
            <span>
              Uncertain between <strong>{GRADES[unsure.first].short}</strong> and{' '}
              <strong>{GRADES[unsure.second].short}</strong>. Review the photograph closely before
              confirming.
            </span>
          </p>
        )}
        {eye.sample && (
          <p className="reference" data-match={eye.sample.label === modelGrade}>
            Reference grade for this test photograph:{' '}
            <strong>{GRADES[asGrade(eye.sample.label)].short}</strong>.{' '}
            {eye.sample.label === modelGrade ? 'The model agrees.' : 'The model disagrees.'}
          </p>
        )}
      </section>

      <Review eye={eye} modelGrade={modelGrade} onReview={onReview} />
    </>
  )
}

/** Probability of each grade, drawn worst grade first so it reads like a severity scale. */
function Ladder({ probs, grade, muted }: { probs: number[]; grade: Grade; muted: boolean }) {
  return (
    <ol className="ladder" data-muted={muted || undefined} aria-label="Probability of each grade">
      {[...GRADE_LIST].reverse().map((g) => {
        const p = probs[g] ?? 0
        return (
          <li key={g} className="rung" data-active={g === grade || undefined}>
            <SeverityPips grade={g} size="sm" />
            <span className="rung-name">
              {GRADES[g].short}
              <span className="rung-num">{g}</span>
            </span>
            <span className="rung-track">
              <span
                className="rung-fill"
                style={{ width: `${Math.max(p * 100, 0.5)}%`, background: GRADES[g].color }}
              />
            </span>
            <span className="rung-pct">{Math.round(p * 100)}%</span>
          </li>
        )
      })}
    </ol>
  )
}

/** Clinician review: confirm model grade or change it with a reason. */
function Review({
  eye,
  modelGrade,
  onReview,
}: {
  eye: EyeCase
  modelGrade: Grade
  onReview: (r: ClinicianReview | null) => void
}) {
  const [editing, setEditing] = useState(false)
  const [grade, setGrade] = useState<Grade>(modelGrade)
  const [reason, setReason] = useState(OVERRIDE_REASONS[0])
  const [note, setNote] = useState('')

  if (eye.review && !editing) {
    const r = eye.review
    return (
      <section className="report-block review-done">
        <h3 className="panel-heading">Clinician review</h3>
        <p>
          {r.decision === 'confirmed' ? 'Confirmed' : 'Changed to'}{' '}
          <strong>{GRADES[r.grade].long.toLowerCase()}</strong>
          {r.decision === 'changed' && <> (model: {GRADES[modelGrade].short})</>}.
        </p>
        {r.decision === 'changed' && <p className="muted">Reason: {r.reason}</p>}
        {r.note && <p className="muted">Note: {r.note}</p>}
        <button type="button" className="btn-link" onClick={() => onReview(null)}>
          Undo review
        </button>
      </section>
    )
  }

  const stamp = () => new Date().toISOString()

  return (
    <section className="report-block review">
      <h3 className="panel-heading">Clinician review</h3>
      {!editing ? (
        <div className="review-actions">
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => onReview({ decision: 'confirmed', grade: modelGrade, reason: '', note: '', at: stamp() })}
          >
            Confirm {GRADES[modelGrade].short}
          </button>
          <button type="button" className="btn" onClick={() => setEditing(true)}>
            Change grade
          </button>
        </div>
      ) : (
        <form
          className="review-form"
          onSubmit={(e) => {
            e.preventDefault()
            onReview({
              decision: grade === modelGrade ? 'confirmed' : 'changed',
              grade,
              reason: grade === modelGrade ? '' : reason,
              note: note.trim(),
              at: stamp(),
            })
            setEditing(false)
          }}
        >
          <fieldset className="grade-picker">
            <legend className="field-label">Your grade</legend>
            {GRADE_LIST.map((g) => (
              <label key={g} className="grade-option">
                <input type="radio" name="grade" checked={grade === g} onChange={() => setGrade(g)} />
                <SeverityPips grade={g} size="sm" showLabel />
              </label>
            ))}
          </fieldset>
          {grade !== modelGrade && (
            <label className="field">
              <span className="field-label">Reason for the change</span>
              <select className="input" value={reason} onChange={(e) => setReason(e.target.value)}>
                {OVERRIDE_REASONS.map((r) => (
                  <option key={r}>{r}</option>
                ))}
              </select>
            </label>
          )}
          <label className="field">
            <span className="field-label">Note (optional)</span>
            <textarea className="input" rows={2} value={note} onChange={(e) => setNote(e.target.value)} maxLength={300} />
          </label>
          <div className="review-actions">
            <button type="submit" className="btn btn-primary">
              Save review
            </button>
            <button type="button" className="btn" onClick={() => setEditing(false)}>
              Cancel
            </button>
          </div>
        </form>
      )}
    </section>
  )
}
