import { useEffect, useRef, useState } from 'react'
import type { KeyboardEvent, ReactNode, SubmitEvent } from 'react'
import { toBlocks } from './answerText'
import {
  ApiRequestError,
  askQuestion,
  fetchHealth,
  validateOfficialUrl,
} from './api'
import type { AskResponse, Citation, HealthResponse, OfficialLink } from './api'
import { initialContext, planMessage } from './conversation'
import type { ConversationContext } from './conversation'

const SAMPLE_QUESTION = "I can't make it on the day I booked. Can I move it?"

interface ChatTurn {
  id: string
  question: string
  response: AskResponse
}

interface PendingFailure {
  displayQuestion: string
  error: ApiRequestError
}

/**
 * Safely render inline **bold** segments and [S1] citation tags as React nodes
 * without using HTML injection.
 */
function renderInlineText(text: string): ReactNode[] {
  const parts = text.split(/(\*\*[^*]+\*\*|\[S\d+\])/g)
  return parts.map((part, idx) => {
    if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
      return <strong key={idx}>{part.slice(2, -2)}</strong>
    }
    if (/^\[S\d+\]$/.test(part)) {
      return (
        <span key={idx} className="cite-badge">
          {part}
        </span>
      )
    }
    return part
  })
}

/**
 * Render answer text with readable paragraphs and numbered steps.
 */
function AnswerBody({ text }: { text: string }) {
  const blocks = toBlocks(text)

  return (
    <div className="answer-body">
      {blocks.map((block, i) =>
        block.kind === 'ol' ? (
          <ol key={i} className="answer-steps">
            {block.items.map((item, j) => (
              <li key={j}>{renderInlineText(item)}</li>
            ))}
          </ol>
        ) : (
          <p key={i}>{renderInlineText(block.text)}</p>
        )
      )}
    </div>
  )
}

function formatPdfPages(pages: number[] | null): string | null {
  if (!pages || pages.length === 0) return null
  if (pages.length === 1) return `PDF page ${pages[0]}`
  return `PDF pages ${pages[0]}–${pages[pages.length - 1]}`
}

function SourcesList({ citations }: { citations: Citation[] }) {
  if (citations.length === 0) return null
  return (
    <div className="sources-section" aria-label="Sources">
      <h3 className="sources-heading">Sources</h3>
      <ul className="sources-list">
        {citations.map((c) => {
          const safeUrl = validateOfficialUrl(c.url)
          const pageLabel = formatPdfPages(c.pdf_pages)
          return (
            <li key={c.id} className="source-item">
              <span className="source-id">[{c.id}]</span>
              <div className="source-details">
                <div className="source-title-row">
                  {safeUrl ? (
                    <a
                      href={safeUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="source-link"
                    >
                      {c.title}
                    </a>
                  ) : (
                    <span className="source-title-text">{c.title}</span>
                  )}
                </div>
                <div className="source-meta" aria-label="Source metadata">
                  <span className="source-chip source-section-chip">
                    {c.section}
                  </span>
                  {pageLabel && (
                    <span className="source-chip source-pages-chip">
                      {pageLabel}
                    </span>
                  )}
                  <span className="source-chip source-date-chip">
                    Reviewed {c.reviewed_on}
                  </span>
                </div>
              </div>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

function OfficialLinksBox({
  links,
  status,
}: {
  links: OfficialLink[]
  status: AskResponse['status']
}) {
  const verified = links
    .map((l) => ({ label: l.label, url: validateOfficialUrl(l.url) }))
    .filter((l): l is { label: string; url: string } => l.url !== null)

  const hadUnverified = links.length > verified.length

  if (verified.length === 0 && !hadUnverified) return null

  return (
    <div className="official-links-box">
      <p className="official-links-title">
        {status === 'official_redirect'
          ? 'Official Passport Seva tool'
          : 'Official Passport Seva reference'}
      </p>
      {verified.length > 0 && (
        <ul className="official-links-list">
          {verified.map((link, idx) => (
            <li key={idx}>
              <a
                href={link.url}
                target="_blank"
                rel="noopener noreferrer"
                className="official-link"
              >
                {link.label} <span aria-hidden="true">↗</span>
              </a>
            </li>
          ))}
        </ul>
      )}
      {hadUnverified && status === 'live_status_redirect' && (
        <p className="unverified-note">
          Direct link omitted because the status-tracker URL could not be verified
          against the live portal right now. Open the official Passport Seva portal
          and select <strong>Track Application Status</strong> on the home page. Do
          not enter passport numbers or application reference numbers here.
        </p>
      )}
    </div>
  )
}

const STATUS_BADGE: Record<
  AskResponse['status'],
  { label: string; tone: 'neutral' | 'warn' | 'info' }
> = {
  answered: { label: 'Grounded in official sources', tone: 'info' },
  clarification: { label: 'Clarification needed', tone: 'warn' },
  official_redirect: { label: 'Official guidance link', tone: 'info' },
  live_status_redirect: { label: 'Live portal required', tone: 'info' },
  conflicting_sources: { label: 'Official sources disagree', tone: 'warn' },
  insufficient_evidence: { label: 'Insufficient indexed evidence', tone: 'neutral' },
  out_of_scope: { label: 'Outside guide scope', tone: 'neutral' },
  error: { label: 'Service error', tone: 'warn' },
}

export function App() {
  const [turns, setTurns] = useState<ChatTurn[]>([])
  const [context, setContext] = useState<ConversationContext>(initialContext)
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null)
  const [failure, setFailure] = useState<PendingFailure | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)

  const inputRef = useRef<HTMLTextAreaElement | null>(null)
  const bottomRef = useRef<HTMLDivElement | null>(null)

  const checkBackendHealth = async () => {
    try {
      const h = await fetchHealth()
      setHealth(h)
      setHealthError(null)
    } catch (err) {
      setHealth(null)
      setHealthError(
        err instanceof ApiRequestError
          ? err.message
          : 'Backend health check failed.'
      )
    }
  }

  useEffect(() => {
    let active = true
    fetchHealth()
      .then((h) => {
        if (!active) return
        setHealth(h)
        setHealthError(null)
      })
      .catch((err: unknown) => {
        if (!active) return
        setHealth(null)
        setHealthError(
          err instanceof ApiRequestError
            ? err.message
            : 'Backend health check failed.'
        )
      })
    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [turns, loading, failure])

  /**
   * Unified handler for both typed submissions and clarification choice buttons.
   */
  const submitUserMessage = async (
    rawMessage: string,
    clearInputOnStart = true
  ) => {
    const trimmed = rawMessage.trim()
    if (!trimmed || loading) return

    const planned = planMessage(trimmed, context)

    if (planned.kind === 'local') {
      const localResponse: AskResponse = {
        status: 'answered',
        answer: planned.answer,
        mode: 'hybrid',
        grounded: false,
        citations: [],
        official_links: [],
        clarification_question: null,
        llm_used: false,
        error: null,
      }
      setTurns((prev) => [
        ...prev,
        {
          id: `${Date.now()}-${prev.length}`,
          question: planned.displayQuestion,
          response: localResponse,
        },
      ])
      setContext(planned.next)
      if (clearInputOnStart) {
        setInput('')
      }
      return
    }

    setLoading(true)
    setPendingQuestion(planned.displayQuestion)
    setFailure(null)

    try {
      const response = await askQuestion(planned.question)
      const nextContext: ConversationContext =
        response.status === 'clarification' && !planned.next.pendingClarification
          ? {
              ...planned.next,
              pendingClarification: {
                originalQuestion: planned.question,
                kind: 'app_type',
              },
            }
          : planned.next

      setTurns((prev) => [
        ...prev,
        {
          id: `${Date.now()}-${prev.length}`,
          question: planned.displayQuestion,
          response,
        },
      ])
      setContext(nextContext)
      if (clearInputOnStart) {
        setInput('')
      }
      if (healthError) {
        void checkBackendHealth()
      }
    } catch (err) {
      const apiErr =
        err instanceof ApiRequestError
          ? err
          : new ApiRequestError(
              'Unexpected error while contacting the server.',
              'service'
            )
      // Preserve the user's question in the input on failure so they can edit or retry
      setInput(planned.displayQuestion)
      setFailure({ displayQuestion: planned.displayQuestion, error: apiErr })
      if (apiErr.kind === 'connection') {
        setHealthError(apiErr.message)
      }
    } finally {
      setLoading(false)
      setPendingQuestion(null)
    }
  }

  const handleFormSubmit = (e: SubmitEvent<HTMLFormElement>) => {
    e.preventDefault()
    void submitUserMessage(input, true)
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      void submitUserMessage(input, true)
    }
  }

  const handleSampleClick = () => {
    setInput(SAMPLE_QUESTION)
    inputRef.current?.focus()
  }

  return (
    <div className="page-shell">
      <header className="site-header">
        <div className="header-inner">
          <div>
            <h1 className="site-title">Passport Seva Application Guide</h1>
            <p className="site-subtitle">
              Independent educational project that answers questions using official
              Passport Seva guidance. Not affiliated with the Government of India.
            </p>
          </div>
          <div className="health-pill-wrap" aria-live="polite">
            {healthError ? (
              <span className="health-pill health-offline" title={healthError}>
                Backend offline
              </span>
            ) : health ? (
              <span
                className={`health-pill ${
                  health.status === 'ok' ? 'health-ok' : 'health-degraded'
                }`}
              >
                {health.status === 'ok'
                  ? `Ready · ${health.index.chunk_count ?? 0} indexed passages`
                  : 'Service degraded'}
              </span>
            ) : (
              <span className="health-pill health-checking">Checking status…</span>
            )}
          </div>
        </div>
      </header>

      <main className="main-layout">
        <section className="chat-column" aria-label="Question and answer">
          <div className="chat-stream">
            {turns.length === 0 && !loading && !failure && (
              <div className="empty-state">
                <p className="empty-intro">
                  Ask a question about applying for an adult ordinary passport in
                  India under the Normal process. Answers cite the official source
                  passages or direct you to the appropriate Passport Seva tool.
                </p>
                <div className="sample-box">
                  <span className="sample-label">Try a sample question:</span>
                  <button
                    type="button"
                    className="sample-button"
                    onClick={handleSampleClick}
                  >
                    {SAMPLE_QUESTION}
                  </button>
                </div>
              </div>
            )}

            {turns.map((turn) => {
              const { response } = turn
              const badge = STATUS_BADGE[response.status]
              return (
                <article key={turn.id} className="turn-group">
                  <div className="message user-message">
                    <div className="message-role">You</div>
                    <p className="message-user-text">{turn.question}</p>
                  </div>

                  <div className="message assistant-message">
                    <div className="assistant-header">
                      <span className="message-role">Guide</span>
                      <span className={`status-tag tone-${badge.tone}`}>
                        {badge.label}
                      </span>
                    </div>

                    <AnswerBody text={response.answer} />

                    {response.status === 'clarification' &&
                      response.clarification_question && (
                        <div className="clarification-box">
                          <p className="clarification-prompt">
                            {response.clarification_question}
                          </p>
                          <div
                            className="clarification-actions"
                            role="group"
                            aria-label="Choose application type to continue"
                          >
                            <button
                              type="button"
                              className="choice-button"
                              disabled={loading}
                              onClick={() =>
                                void submitUserMessage('Fresh', false)
                              }
                            >
                              Fresh application (first ordinary passport)
                            </button>
                            <button
                              type="button"
                              className="choice-button"
                              disabled={loading}
                              onClick={() =>
                                void submitUserMessage('Reissue', false)
                              }
                            >
                              Reissue (previously held an ordinary passport)
                            </button>
                          </div>
                        </div>
                      )}

                    <OfficialLinksBox
                      links={response.official_links}
                      status={response.status}
                    />

                    <SourcesList citations={response.citations} />
                  </div>
                </article>
              )
            })}

            {loading && pendingQuestion && (
              <div
                className="turn-group"
                role="status"
                aria-live="polite"
                aria-busy="true"
              >
                <div className="message user-message">
                  <div className="message-role">You</div>
                  <p className="message-user-text">{pendingQuestion}</p>
                </div>
                <div className="message assistant-message loading-message">
                  <span className="spinner" aria-hidden="true" />
                  <span>Searching official sources and preparing answer…</span>
                </div>
              </div>
            )}

            {failure && (
              <div className="error-banner" role="alert" aria-live="assertive">
                <div className="error-banner-header">
                  <strong>
                    {failure.error.kind === 'connection'
                      ? 'Backend connection failed'
                      : failure.error.kind === 'rate_limited'
                      ? 'Rate limit reached'
                      : failure.error.kind === 'timeout'
                      ? 'Request timed out'
                      : failure.error.kind === 'validation'
                      ? 'Invalid question'
                      : 'Service error'}
                  </strong>
                </div>
                <p className="error-banner-text">{failure.error.message}</p>
                <div className="error-banner-actions">
                  <button
                    type="button"
                    className="retry-button"
                    disabled={loading}
                    onClick={() =>
                      void submitUserMessage(failure.displayQuestion, true)
                    }
                  >
                    Retry question
                  </button>
                </div>
              </div>
            )}

            <div ref={bottomRef} />
          </div>

          <form className="composer" onSubmit={handleFormSubmit} noValidate>
            <div className="composer-label-row">
              <label htmlFor="question-input" className="composer-label">
                Your question
              </label>
              <span className="composer-hint">
                Do not enter passport numbers, ARNs, or personal documents.
              </span>
            </div>
            <div className="composer-controls">
              <textarea
                id="question-input"
                ref={inputRef}
                className="composer-textarea"
                rows={2}
                maxLength={500}
                value={input}
                disabled={loading}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about fresh applications, routine reissues, or appointments…"
              />
              <button
                type="submit"
                className="send-button"
                disabled={loading || !input.trim()}
              >
                {loading ? 'Sending…' : 'Send'}
              </button>
            </div>
            <div className="composer-footer">
              <span>Press Enter to send, Shift+Enter for a new line.</span>
              <span>{input.length}/500</span>
            </div>
          </form>
        </section>

        <aside className="info-panel" aria-label="Supported scope and guidance">
          <div className="info-card">
            <h2 className="info-heading">Supported topics</h2>
            <p className="info-lead">
              Covers adults in India applying for an ordinary passport through the{' '}
              <strong>Normal</strong> process:
            </p>
            <ul className="info-list">
              <li>
                <strong>Fresh applications</strong> — portal registration, filling
                and submitting the form, and Application Reference Number (ARN).
              </li>
              <li>
                <strong>Routine reissues</strong> — when to choose Reissue versus
                Fresh, and routing to the official Document Advisor.
              </li>
              <li>
                <strong>Appointments</strong> — paying and booking a slot at a PSK or
                POPSK, rescheduling an appointment, and bringing originals with
                self-attested copies.
              </li>
            </ul>
          </div>

          <div className="info-card">
            <h2 className="info-heading">How answers work</h2>
            <ul className="info-list">
              <li>
                Answers use only the indexed official Passport Seva guides, FAQs,
                and the appointment booking PDF.
              </li>
              <li>
                For case-specific document checklists or live dispatch status, the
                guide directs you to the official Passport Seva portal instead of
                guessing.
              </li>
              <li>
                Where official sources disagree on the number of allowed
                reschedules, the guide explains the discrepancy and directs you to
                check the attempts shown in your account.
              </li>
            </ul>
          </div>
        </aside>
      </main>
    </div>
  )
}

export default App
