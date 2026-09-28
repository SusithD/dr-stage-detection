/**
 * Small wrapper around the backend API. Errors come back as readable messages for the UI.
 */
import type { HealthResponse, PredictResponse, Sample } from './types'

const API_BASE = '/api'

export const SERVER_UNREACHABLE =
  "Can't reach the RetinaScan server. Start it with: uv run python -m drstage.cli.serve"

/** Fetch JSON from the API and extract server error detail if the response fails. */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, init)
  } catch {
    throw new Error(SERVER_UNREACHABLE)
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    const detail = body && typeof body.detail === 'string' ? body.detail : null
    throw new Error(detail ?? `The server returned an error (${response.status}). Try again.`)
  }
  return response.json() as Promise<T>
}

/** Endpoint functions: health, samples gallery, model prediction, and image blob fetch. */
export const api = {
  health: () => request<HealthResponse>('/health'),
  samples: () => request<Sample[]>('/samples'),
  predict: (file: Blob, filename: string) => {
    const body = new FormData()
    body.append('file', file, filename)
    return request<PredictResponse>('/predict', { method: 'POST', body })
  },
  async sampleImage(id: string): Promise<Blob> {
    const response = await fetch(`${API_BASE}/samples/${encodeURIComponent(id)}/image`)
    if (!response.ok) throw new Error('This sample image could not be loaded. Choose another.')
    return response.blob()
  },
}
