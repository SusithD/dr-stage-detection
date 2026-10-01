import { asGrade, visitOutcome } from './grades'
import type { EyeCase, EyeSide, VisitContext } from './types'

/** Suggested prompts for empty state. */
export const SUGGESTED_PROMPTS = [
  'Explain this result',
  'How confident is the model?',
  'Is the image quality good enough?',
  'Draft a referral letter',
]

/** Map backend tool names to friendly labels. */
export const TOOL_LABELS: Record<string, string> = {
  get_visit_summary: 'Visit summary',
  get_eye_details: 'Eye details',
  lookup_guideline: 'Grading guidelines',
}

/** Build VisitContext from the current eye cases. Only includes eyes with done status and a result. */
export function buildVisitContext(eyes: Record<EyeSide, EyeCase | null>): VisitContext {
  const doneEyes = (['right', 'left'] as EyeSide[])
    .map((side) => eyes[side])
    .filter((e): e is EyeCase => e !== null && e.status === 'done' && e.result !== null)

  const outcome = visitOutcome(doneEyes)

  const eyeContexts = doneEyes.map((eye) => {
    const result = eye.result!
    return {
      side: eye.side,
      grade: asGrade(result.grade),
      grade_name: result.grade_name,
      class_probs: result.class_probs,
      prob_dr: result.prob_dr,
      referable: result.referable,
      gradable: result.quality.gradable,
      checks: result.quality.checks.map((c) => ({
        label: c.label,
        status: c.status,
        message: c.message,
      })),
      attention: result.attention,
      reference_grade: eye.sample ? asGrade(eye.sample.label) : null,
      review: eye.review
        ? {
            decision: eye.review.decision,
            grade: eye.review.grade,
            reason: eye.review.reason,
            note: eye.review.note,
          }
        : null,
    }
  })

  return {
    eyes: eyeContexts,
    outcome_title: outcome.title,
    outcome_detail: outcome.detail,
  }
}
