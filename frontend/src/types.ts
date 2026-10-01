/**
 * Types for DR screening: ICDR grades 0-4, eye sides, viewer modes,
 * API responses, clinician review, and image quality checks.
 */

export type Grade = 0 | 1 | 2 | 3 | 4
export type EyeSide = 'right' | 'left'
export type ViewMode = 'original' | 'enhanced' | 'compare' | 'heatmap'
export type CheckStatus = 'ok' | 'warn' | 'fail'

export interface ModelInfo {
  name: string
  head: string
  image_size: number
  preprocessing: string
  checkpoint: string
}

export interface HealthResponse {
  status: string
  demo: boolean
  device: string
  model: ModelInfo | null
  assistant: boolean
}

export interface QualityCheck {
  key: string
  label: string
  status: CheckStatus
  value: number
  message: string
}

export interface Attention {
  region: string
  spread: number
}

export interface PredictResponse {
  demo: boolean
  grade: number
  grade_name: string
  dr_detected: boolean
  prob_dr: number
  referable: boolean
  class_probs: number[]
  images: { original: string; enhanced: string; heatmap: string }
  quality: { gradable: boolean; checks: QualityCheck[] }
  inference_ms: number
  attention: Attention
}

export interface Sample {
  id: string
  label: number
  grade_name: string
  thumbnail: string
}

export interface ClinicianReview {
  decision: 'confirmed' | 'changed'
  grade: Grade
  reason: string
  note: string
  at: string
}

export interface EyeCase {
  side: EyeSide
  filename: string
  preview: string
  sample: Sample | null
  status: 'analysing' | 'done' | 'error'
  result: PredictResponse | null
  error: string | null
  review: ClinicianReview | null
}

export interface AssistantMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface AssistantReply {
  reply: string
  tools_used: string[]
}

export interface EyeContext {
  side: EyeSide
  grade: Grade
  grade_name: string
  class_probs: number[]
  prob_dr: number
  referable: boolean
  gradable: boolean
  checks: { label: string; status: CheckStatus; message: string }[]
  attention: Attention
  reference_grade: Grade | null
  review: { decision: 'confirmed' | 'changed'; grade: Grade; reason: string; note: string } | null
}

export interface VisitContext {
  eyes: EyeContext[]
  outcome_title: string
  outcome_detail: string
}
