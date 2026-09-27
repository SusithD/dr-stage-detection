import type { EyeCase, Grade, PredictResponse } from './types'

export interface GradeInfo {
  short: string
  long: string
  color: string
}

export const GRADES: Record<Grade, GradeInfo> = {
  0: { short: 'No DR', long: 'No apparent retinopathy', color: 'var(--g0)' },
  1: { short: 'Mild', long: 'Mild non-proliferative retinopathy', color: 'var(--g1)' },
  2: { short: 'Moderate', long: 'Moderate non-proliferative retinopathy', color: 'var(--g2)' },
  3: { short: 'Severe', long: 'Severe non-proliferative retinopathy', color: 'var(--g3)' },
  4: { short: 'Proliferative', long: 'Proliferative retinopathy', color: 'var(--g4)' },
}

export const GRADE_LIST: Grade[] = [0, 1, 2, 3, 4]

export const PREPROCESSING_NAMES: Record<string, string> = {
  ben_graham: 'Ben Graham local colour normalisation',
  clahe_l: 'CLAHE on lightness',
  none: 'Circle crop only',
}

export const OVERRIDE_REASONS = [
  'Lesions missed by the model',
  'Model over-graded normal features',
  'Image quality limits grading',
  'Clinical history differs',
  'Other',
]

export function asGrade(value: number): Grade {
  return Math.min(4, Math.max(0, Math.round(value))) as Grade
}

/** Grade used for decisions: the clinician's grade when reviewed, otherwise the model's. */
export function finalGrade(eye: EyeCase): Grade | null {
  if (eye.review) return eye.review.grade
  return eye.result ? asGrade(eye.result.grade) : null
}

export interface Uncertainty {
  first: Grade
  second: Grade
  margin: number
}

/** Flags predictions whose top two grades are close or whose top grade is weak. */
export function uncertainty(result: PredictResponse): Uncertainty | null {
  const ranked = result.class_probs
    .map((p, g) => ({ p, g: g as Grade }))
    .sort((a, b) => b.p - a.p)
  const margin = ranked[0].p - ranked[1].p
  if (margin >= 0.2 && ranked[0].p >= 0.5) return null
  const [a, b] = [ranked[0].g, ranked[1].g].sort((x, y) => x - y)
  return { first: a, second: b, margin }
}

export interface Outcome {
  tone: 'routine' | 'watch' | 'refer' | 'urgent' | 'retake' | 'pending'
  title: string
  detail: string
}

/** Visit-level screening outcome: the worse eye decides, following common screening practice. */
export function visitOutcome(eyes: EyeCase[]): Outcome {
  const done = eyes.filter((e) => e.status === 'done' && e.result)
  if (done.length === 0) {
    return {
      tone: 'pending',
      title: 'Awaiting photographs',
      detail: 'Add a photograph of each eye, or choose a sample from the test set.',
    }
  }
  const gradable = done.filter((e) => e.result!.quality.gradable || e.review)
  const ungradable = done.filter((e) => !e.result!.quality.gradable && !e.review)
  const worst = Math.max(-1, ...gradable.map((e) => finalGrade(e) ?? -1))

  if (worst >= 4) {
    return {
      tone: 'urgent',
      title: 'Urgent referral',
      detail: 'Proliferative retinopathy can threaten sight. Referral within 1 week is typical.',
    }
  }
  if (worst === 3) {
    return {
      tone: 'refer',
      title: 'Refer to an eye specialist',
      detail: 'Severe non-proliferative retinopathy. Referral within 4 weeks is typical.',
    }
  }
  if (worst === 2) {
    return {
      tone: 'refer',
      title: 'Refer to an eye specialist',
      detail: 'Moderate retinopathy is referable. Referral within 3 months is typical.',
    }
  }
  if (ungradable.length > 0) {
    return {
      tone: 'retake',
      title: 'Retake photograph',
      detail: `The ${ungradable.map((e) => `${e.side} eye`).join(' and ')} photograph failed the quality check, so it can't be graded reliably.`,
    }
  }
  if (worst === 1) {
    return {
      tone: 'watch',
      title: 'Re-screen sooner',
      detail: 'Mild retinopathy only. Re-screening in 6 to 12 months is typical.',
    }
  }
  return {
    tone: 'routine',
    title: 'Routine re-screening',
    detail: 'No retinopathy found. Re-screening in 12 months is typical.',
  }
}
