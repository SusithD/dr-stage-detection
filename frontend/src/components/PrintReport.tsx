/**
 * A4 referral report. It stays hidden on screen and only appears when the page is printed
 * (Print report button or Ctrl+P).
 */
import { asGrade, finalGrade, GRADES, visitOutcome } from '../grades'
import type { EyeCase, EyeSide, HealthResponse } from '../types'

interface PrintReportProps {
  patientRef: string
  eyes: Record<EyeSide, EyeCase | null>
  health: HealthResponse | null
}

export function PrintReport({ patientRef, eyes, health }: PrintReportProps) {
  const list = [eyes.right, eyes.left].filter((e): e is EyeCase => e !== null)
  const outcome = visitOutcome(list)
  const printed = new Date().toLocaleString(undefined, { dateStyle: 'long', timeStyle: 'short' })

  return (
    <article className="print-report" aria-hidden="true">
      <header className="pr-head">
        <div>
          <h1>Diabetic retinopathy screening report</h1>
          <p>RetinaScan research prototype</p>
        </div>
        <dl>
          <dt>Patient reference</dt>
          <dd>{patientRef || 'Not recorded'}</dd>
          <dt>Date</dt>
          <dd>{printed}</dd>
        </dl>
      </header>

      <section className="pr-outcome">
        <h2>{outcome.title}</h2>
        <p>{outcome.detail}</p>
      </section>

      <section className="pr-eyes">
        {(['right', 'left'] as EyeSide[]).map((side) => {
          const e = eyes[side]
          const g = e ? finalGrade(e) : null
          return (
            <div key={side} className="pr-eye">
              <h3>{side === 'right' ? 'Right eye (OD)' : 'Left eye (OS)'}</h3>
              {e?.result ? (
                <>
                  <div className="pr-images">
                    <img src={e.result.images.original} alt="" />
                    <div className="pr-heat">
                      <img src={e.result.images.original} alt="" />
                      <img src={e.result.images.heatmap} alt="" style={{ opacity: 0.6 }} />
                    </div>
                  </div>
                  <table>
                    <tbody>
                      <tr>
                        <th>Final grade</th>
                        <td>
                          {g !== null ? `${GRADES[g].long} (ICDR ${g})` : 'n/a'}
                          {e.review ? ', clinician reviewed' : ', not yet reviewed'}
                        </td>
                      </tr>
                      <tr>
                        <th>Model grade</th>
                        <td>
                          {GRADES[asGrade(e.result.grade)].short}, any-retinopathy probability{' '}
                          {Math.round(e.result.prob_dr * 100)}%
                        </td>
                      </tr>
                      <tr>
                        <th>Image quality</th>
                        <td>
                          {e.result.quality.gradable ? 'Gradable' : 'Ungradable'}
                          {e.result.quality.checks
                            .filter((c) => c.status !== 'ok')
                            .map((c) => `; ${c.label.toLowerCase()}: ${c.message}`)
                            .join('')}
                        </td>
                      </tr>
                      {e.review?.decision === 'changed' && (
                        <tr>
                          <th>Reason for change</th>
                          <td>{e.review.reason}</td>
                        </tr>
                      )}
                      {e.review?.note && (
                        <tr>
                          <th>Clinician note</th>
                          <td>{e.review.note}</td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </>
              ) : (
                <p>Not photographed or not analysed.</p>
              )}
            </div>
          )
        })}
      </section>

      <footer className="pr-foot">
        <p>
          Model: {health?.model?.name ?? 'demo mode, simulated results'}. Heatmaps show the regions
          that most influenced the model and are not lesion annotations.
        </p>
        <p>
          Research prototype for a university coursework project. Not a medical device. A qualified
          clinician must confirm every result. Referral timeframes are typical values; follow local
          protocols.
        </p>
        <p className="pr-sign">Reviewed by: ______________________ Signature: ______________</p>
      </footer>
    </article>
  )
}
