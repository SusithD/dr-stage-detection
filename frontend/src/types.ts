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
}

export interface QualityCheck {
  key: string
  label: string
  status: CheckStatus
  value: number
  message: string
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
