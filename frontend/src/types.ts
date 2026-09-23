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

export interface PredictImages {
  original: string
  enhanced: string
  heatmap: string
}

export interface PredictResponse {
  demo: boolean
  grade: number
  grade_name: string
  dr_detected: boolean
  prob_dr: number
  referable: boolean
  class_probs: number[]
  images: PredictImages
  inference_ms: number
}

export interface Sample {
  id: string
  label: number
  grade_name: string
  thumbnail: string
}

export type Grade = 0 | 1 | 2 | 3 | 4

export type ViewMode = 'original' | 'enhanced' | 'heatmap'
