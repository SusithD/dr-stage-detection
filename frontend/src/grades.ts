import type { Grade } from './types'

export interface GradeInfo {
  short: string
  long: string
  guidance: string
  color: string
  onColor: string
}

export const GRADES: Record<Grade, GradeInfo> = {
  0: {
    short: 'No DR',
    long: 'No apparent retinopathy',
    guidance: 'No signs of retinopathy found. Routine screening in 12 months is typical.',
    color: 'var(--g0)',
    onColor: '#ffffff',
  },
  1: {
    short: 'Mild',
    long: 'Mild non-proliferative retinopathy',
    guidance: 'Early changes found. Re-screening sooner, often in 6–12 months, is typical.',
    color: 'var(--g1)',
    onColor: 'var(--ink)',
  },
  2: {
    short: 'Moderate',
    long: 'Moderate non-proliferative retinopathy',
    guidance: 'Referable retinopathy. Referral to an eye specialist is typical.',
    color: 'var(--g2)',
    onColor: 'var(--ink)',
  },
  3: {
    short: 'Severe',
    long: 'Severe non-proliferative retinopathy',
    guidance: 'Severe retinopathy. Prompt referral to an eye specialist is typical.',
    color: 'var(--g3)',
    onColor: '#ffffff',
  },
  4: {
    short: 'Proliferative',
    long: 'Proliferative retinopathy',
    guidance: 'Proliferative retinopathy, which can threaten sight. Urgent referral is typical.',
    color: 'var(--g4)',
    onColor: '#ffffff',
  },
}

export const GRADE_LIST: Grade[] = [0, 1, 2, 3, 4]

export const DISCLAIMER = 'A qualified clinician must confirm any result.'

export const PREPROCESSING_NAMES: Record<string, string> = {
  ben_graham: 'Ben Graham local colour normalisation',
  clahe_l: 'CLAHE on lightness',
  none: 'None',
}

export function asGrade(value: number): Grade {
  return Math.min(4, Math.max(0, Math.round(value))) as Grade
}
