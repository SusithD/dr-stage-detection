import { asGrade, DISCLAIMER, GRADES, PREPROCESSING_NAMES } from '../grades'
import type { HealthResponse, PredictResponse } from '../types'
import { SeverityScale } from './SeverityScale'

interface FindingsProps {
  result: PredictResponse
  health: HealthResponse | null
}

export function Findings({ result, health }: FindingsProps) {
  const grade = asGrade(result.grade)
  const info = GRADES[grade]
  const model = health?.model

  return (
    <article className="findings">
      <header className="verdict" style={{ borderColor: info.color }}>
        <h2 className="verdict-title">
          {result.dr_detected
            ? 'Diabetic retinopathy detected'
            : 'No diabetic retinopathy detected'}
        </h2>
        <p className="verdict-stage">
          {info.long} (grade {grade})
        </p>
        <p className="verdict-confidence">
          Model confidence for any retinopathy: {Math.round(result.prob_dr * 100)}%
        </p>
      </header>

      <SeverityScale grade={grade} classProbs={result.class_probs} />

      <section className="guidance">
        <h3>What this means</h3>
        <p>{info.guidance}</p>
        <p className="guidance-caveat">{DISCLAIMER}</p>
      </section>

      <details className="model-details">
        <summary>Model details</summary>
        <dl>
          {model ? (
            <>
              <dt>Network</dt>
              <dd>{model.name}</dd>
              <dt>Output head</dt>
              <dd>{model.head === 'corn' ? 'Ordinal (CORN)' : 'Softmax classifier'}</dd>
              <dt>Input size</dt>
              <dd>
                {model.image_size} × {model.image_size} px
              </dd>
              <dt>Preprocessing</dt>
              <dd>{PREPROCESSING_NAMES[model.preprocessing] ?? model.preprocessing}</dd>
            </>
          ) : (
            <>
              <dt>Network</dt>
              <dd>None loaded (demo mode)</dd>
            </>
          )}
          <dt>Device</dt>
          <dd>{health?.device ?? 'unknown'}</dd>
          <dt>Analysis time</dt>
          <dd>{Math.round(result.inference_ms)} ms</dd>
        </dl>
      </details>
    </article>
  )
}
