/**
 * API client and types matching backend/app/schemas.py and backend/app/main.py.
 * Uses relative /api paths (proxied by Vite in local development to http://127.0.0.1:8000).
 */

export type Mode = 'semantic' | 'hybrid' | 'rerank'

/**
 * Backend base. VITE_API_BASE_URL is an ORIGIN only (e.g. http://192.168.0.106:8000),
 * without a trailing /api: the client appends /api/health and /api/ask itself.
 * A trailing slash or a trailing /api is tolerated and stripped. When unset
 * (browser dev), relative /api URLs go through the Vite proxy.
 */
export function apiUrl(path: string): string {
  const base = (import.meta.env.VITE_API_BASE_URL ?? '')
    .trim()
    .replace(/\/+$/, '')
    .replace(/\/api$/i, '')
  return `${base}${path}`
}

export type AskStatus =
  | 'answered'
  | 'insufficient_evidence'
  | 'clarification'
  | 'official_redirect'
  | 'live_status_redirect'
  | 'conflicting_sources'
  | 'out_of_scope'
  | 'error'

export type ErrorCode =
  | 'groq_not_configured'
  | 'groq_auth_rejected'
  | 'groq_rate_limited'
  | 'groq_timeout'
  | 'groq_unavailable'
  | 'index_unavailable'
  | 'retrieval_failed'

export interface AskRequest {
  question: string
  mode?: Mode
}

export interface Citation {
  id: string
  title: string
  section: string
  url: string
  pdf_pages: number[] | null
  source_id: string
  reviewed_on: string
  evidence_role: 'answer_evidence' | 'routing_only'
}

export interface OfficialLink {
  label: string
  url: string
}

export interface ErrorInfo {
  code: ErrorCode
  message: string
}

export interface AskResponse {
  status: AskStatus
  answer: string
  mode: Mode
  grounded: boolean
  citations: Citation[]
  official_links: OfficialLink[]
  clarification_question: string | null
  llm_used: boolean
  error: ErrorInfo | null
}

export interface IndexHealth {
  ready: boolean
  chunk_count: number | null
  embedding_model: string | null
  similarity_metric: string | null
  detail: string | null
}

export interface GroqHealth {
  configured: boolean
  model: string
}

export interface HealthResponse {
  status: 'ok' | 'degraded'
  backend: 'up'
  index: IndexHealth
  groq: GroqHealth
}

export type RequestErrorKind =
  | 'connection'
  | 'rate_limited'
  | 'timeout'
  | 'validation'
  | 'service'

export class ApiRequestError extends Error {
  readonly kind: RequestErrorKind
  readonly code?: ErrorCode
  readonly httpStatus?: number

  constructor(
    message: string,
    kind: RequestErrorKind,
    options?: { code?: ErrorCode; httpStatus?: number }
  ) {
    super(message)
    this.name = 'ApiRequestError'
    this.kind = kind
    this.code = options?.code
    this.httpStatus = options?.httpStatus
  }
}

/**
 * The configured live-status tracker URL on the backend returned HTTP 503 / 404
 * when probed and is not part of source_manifest.json, so it cannot be verified
 * right now. We omit unverified tracker paths from clickable links in the UI.
 */
const UNVERIFIED_PATHS = ['/AppOnlineProject/statusTracker/trackStatusInpNew']

/**
 * Validate that a URL is HTTPS on passportindia.gov.in (or a subdomain)
 * and is not on the unverified path list. Preserves #page=N fragments.
 */
export function validateOfficialUrl(rawUrl: string): string | null {
  try {
    const parsed = new URL(rawUrl)
    if (parsed.protocol !== 'https:') {
      return null
    }
    const host = parsed.hostname.toLowerCase()
    if (host !== 'passportindia.gov.in' && !host.endsWith('.passportindia.gov.in')) {
      return null
    }
    if (UNVERIFIED_PATHS.some((p) => parsed.pathname === p)) {
      return null
    }
    return parsed.toString()
  } catch {
    return null
  }
}

export async function fetchHealth(): Promise<HealthResponse> {
  let res: Response
  try {
    res = await fetch(apiUrl('/api/health'), {
      method: 'GET',
      headers: { Accept: 'application/json' },
    })
  } catch {
    throw new ApiRequestError(
      'Cannot reach the backend server. Check that FastAPI is running on http://127.0.0.1:8000.',
      'connection'
    )
  }
  if (!res.ok) {
    throw new ApiRequestError(
      `Health check returned HTTP ${res.status}.`,
      'service',
      { httpStatus: res.status }
    )
  }
  return (await res.json()) as HealthResponse
}

export async function askQuestion(question: string): Promise<AskResponse> {
  const payload: AskRequest = { question, mode: 'hybrid' }
  let res: Response
  try {
    res = await fetch(apiUrl('/api/ask'), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify(payload),
    })
  } catch {
    throw new ApiRequestError(
      'Could not connect to the backend server. Make sure FastAPI is running and try again.',
      'connection'
    )
  }

  let data: unknown = null
  try {
    data = await res.json()
  } catch {
    // Non-JSON error response (e.g. Vite proxy 502/504 when backend is down)
    throw new ApiRequestError(
      'Could not reach the backend server. Make sure FastAPI is running and try again.',
      'connection',
      { httpStatus: res.status }
    )
  }

  if (res.status === 422) {
    const detail =
      typeof data === 'object' && data !== null && 'detail' in data
        ? String((data as { detail: unknown }).detail)
        : 'Please enter a valid question (up to 500 characters).'
    throw new ApiRequestError(detail, 'validation', { httpStatus: 422 })
  }

  const askRes = data as AskResponse

  if (!res.ok || askRes.status === 'error') {
    const code = askRes?.error?.code
    const msg = askRes?.error?.message || askRes?.answer || `Service error (HTTP ${res.status}).`
    if (res.status === 429 || code === 'groq_rate_limited') {
      throw new ApiRequestError(
        'The answer service is rate-limited right now. Please wait a moment and retry.',
        'rate_limited',
        { code, httpStatus: res.status }
      )
    }
    if (res.status === 504 || code === 'groq_timeout') {
      throw new ApiRequestError(
        'The answer generator timed out. Please retry your question.',
        'timeout',
        { code, httpStatus: res.status }
      )
    }
    throw new ApiRequestError(msg, 'service', { code, httpStatus: res.status })
  }

  return askRes
}

