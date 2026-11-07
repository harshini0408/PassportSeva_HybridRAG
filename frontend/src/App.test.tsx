import '@testing-library/jest-dom/vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { apiUrl, validateOfficialUrl } from './api'
import type { AskResponse, HealthResponse } from './api'
import { initialContext, planMessage } from './conversation'

const HEALTH_OK: HealthResponse = {
  status: 'ok',
  backend: 'up',
  index: {
    ready: true,
    chunk_count: 18,
    embedding_model: 'sentence-transformers/all-MiniLM-L6-v2',
    similarity_metric: 'cosine (inner product over L2-normalised embeddings)',
    detail: null,
  },
  groq: {
    configured: true,
    model: 'openai/gpt-oss-20b',
  },
}

beforeEach(() => {
  Element.prototype.scrollIntoView = vi.fn()
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('apiUrl', () => {
  afterEach(() => vi.unstubAllEnvs())

  it('is relative when VITE_API_BASE_URL is unset and never duplicates /api', () => {
    vi.stubEnv('VITE_API_BASE_URL', '')
    expect(apiUrl('/api/ask')).toBe('/api/ask')
    vi.stubEnv('VITE_API_BASE_URL', 'http://192.168.0.106:8000/')
    expect(apiUrl('/api/ask')).toBe('http://192.168.0.106:8000/api/ask')
    vi.stubEnv('VITE_API_BASE_URL', 'https://example.com/api')
    expect(apiUrl('/api/health')).toBe('https://example.com/api/health')
  })
})

describe('validateOfficialUrl', () => {
  it('accepts HTTPS URLs on passportindia.gov.in and subdomains and preserves #page fragments', () => {
    expect(
      validateOfficialUrl(
        'https://passportindia.gov.in/AppOnlineProject/pdf/New_Online_Appointment_Booking_Process.pdf#page=5'
      )
    ).toBe(
      'https://passportindia.gov.in/AppOnlineProject/pdf/New_Online_Appointment_Booking_Process.pdf#page=5'
    )
    expect(
      validateOfficialUrl(
        'https://services2.passportindia.gov.in/psp/docAdvisor/reissuePassport'
      )
    ).toBe('https://services2.passportindia.gov.in/psp/docAdvisor/reissuePassport')
  })

  it('rejects non-HTTPS, external domains, and the unverified tracker path', () => {
    expect(validateOfficialUrl('http://passportindia.gov.in/psp/apply')).toBeNull()
    expect(validateOfficialUrl('https://evil.example.com/psp/apply')).toBeNull()
    expect(
      validateOfficialUrl(
        'https://passportindia.gov.in/AppOnlineProject/statusTracker/trackStatusInpNew'
      )
    ).toBeNull()
  })
})

describe('Conversation context planner regressions', () => {
  it('responds locally to standalone "okay thank you" without clearing context, while passing real questions through', () => {
    const first = planMessage('How do I renew my passport?', initialContext())
    expect(first.kind).toBe('ask')
    expect(first.next.appType).toBe('reissue')
    expect(first.next.topic).toBe('application')

    const ack = planMessage('okay thank you', first.next)
    expect(ack.kind).toBe('local')
    expect(ack.next.appType).toBe('reissue')
    expect(ack.next.topic).toBe('application')

    const ackWithQuestion = planMessage(
      'Okay thank you, what documents do I need?',
      ack.next
    )
    expect(ackWithQuestion.kind).toBe('ask')
    if (ackWithQuestion.kind === 'ask') {
      expect(ackWithQuestion.question).toBe(
        'For a Normal adult Reissue passport application: what documents do I need?'
      )
    }
  })

  it('handles renewal -> which link and renewal -> thank you -> documents -> topic change -> type override', () => {
    // 1. renewal -> which link
    const step1 = planMessage('How do I renew my passport?', initialContext())
    const linkStep = planMessage('Which link should I use?', step1.next)
    expect(linkStep.kind).toBe('ask')
    if (linkStep.kind === 'ask') {
      expect(linkStep.question).toBe(
        'Which official link should I use to apply for a Reissue ordinary passport?'
      )
    }

    // 2. renewal -> thank you -> documents (does not ask Fresh vs Reissue again)
    const afterAck = planMessage('okay thank you', step1.next)
    expect(afterAck.kind).toBe('local')
    const docsFollowUp = planMessage('What documents do I need?', afterAck.next)
    expect(docsFollowUp.kind).toBe('ask')
    if (docsFollowUp.kind === 'ask') {
      expect(docsFollowUp.question).toBe(
        'For a Normal adult Reissue passport application: What documents do I need?'
      )
      expect(docsFollowUp.next.pendingClarification).toBeNull()
    }

    // 3. change of topic (appointments) does not prepend document context
    const topicChange = planMessage(
      'How do I book my appointment?',
      docsFollowUp.next
    )
    expect(topicChange.kind).toBe('ask')
    if (topicChange.kind === 'ask') {
      expect(topicChange.question).toBe('How do I book my appointment?')
      expect(topicChange.next.topic).toBe('appointment')
    }

    // 4. explicit application type in a new question overrides the remembered one
    const override = planMessage(
      'What documents do I need for a first time fresh passport?',
      topicChange.next
    )
    expect(override.kind).toBe('ask')
    if (override.kind === 'ask') {
      expect(override.question).toBe(
        'What documents do I need for a first time fresh passport?'
      )
      expect(override.next.appType).toBe('fresh')
    }
  })

  it('preserves pending document question whether the user types "reissue" or clicks the button', () => {
    const docQ = planMessage('What exact documents do I need?', initialContext())
    expect(docQ.next.pendingClarification).toEqual({
      originalQuestion: 'What exact documents do I need?',
      kind: 'app_type',
    })

    const typedReply = planMessage('reissue', docQ.next)
    expect(typedReply.kind).toBe('ask')
    if (typedReply.kind === 'ask') {
      expect(typedReply.question).toBe(
        'For a Normal adult Reissue passport application: What exact documents do I need?'
      )
      expect(typedReply.next.appType).toBe('reissue')
      expect(typedReply.next.pendingClarification).toBeNull()
    }
  })
})

describe('Passport Seva Application Guide UI', () => {
  it('renders header, supported topics panel, and fills + focuses input when sample question is clicked', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
      new Response(JSON.stringify(HEALTH_OK), { status: 200 })
    )
    const user = userEvent.setup()
    render(<App />)

    expect(
      screen.getByRole('heading', { name: 'Passport Seva Application Guide' })
    ).toBeInTheDocument()
    expect(
      screen.getByText(/Independent educational project/i)
    ).toBeInTheDocument()
    expect(
      screen.getByRole('heading', { name: 'Supported topics' })
    ).toBeInTheDocument()

    const sampleBtn = screen.getByRole('button', {
      name: "I can't make it on the day I booked. Can I move it?",
    })
    await user.click(sampleBtn)

    const textarea = screen.getByLabelText('Your question') as HTMLTextAreaElement
    expect(textarea.value).toBe(
      "I can't make it on the day I booked. Can I move it?"
    )
    expect(document.activeElement).toBe(textarea)
  })

  it('flow (a): renders grounded reschedule answer with numbered steps and separated source chips preserving #page=5', async () => {
    const answerPayload: AskResponse = {
      status: 'answered',
      answer:
        'Yes, you can move your appointment: 1. Open Submitted Applications and click **View Form**. 2. Click **Reschedule Appointment/Cancel Appointment** [S1].',
      mode: 'hybrid',
      grounded: true,
      citations: [
        {
          id: 'S1',
          title: 'New Process for Online Appointment Booking',
          section: '3. Cancel/Reschedule Appointment',
          url: 'https://passportindia.gov.in/AppOnlineProject/pdf/New_Online_Appointment_Booking_Process.pdf#page=5',
          pdf_pages: [5, 6],
          source_id: 'appointment',
          reviewed_on: '2026-11-06',
          evidence_role: 'answer_evidence',
        },
      ],
      official_links: [],
      clarification_question: null,
      llm_used: true,
      error: null,
    }

    vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(new Response(JSON.stringify(HEALTH_OK), { status: 200 }))
      .mockResolvedValueOnce(
        new Response(JSON.stringify(answerPayload), { status: 200 })
      )

    const user = userEvent.setup()
    render(<App />)

    await user.click(
      screen.getByRole('button', {
        name: "I can't make it on the day I booked. Can I move it?",
      })
    )
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Sources' })).toBeInTheDocument()
    )
    expect(screen.getAllByRole('listitem').length).toBeGreaterThanOrEqual(2)
    expect(screen.getByText('3. Cancel/Reschedule Appointment')).toBeInTheDocument()
    expect(screen.getByText('PDF pages 5–6')).toBeInTheDocument()
    expect(screen.getByText('Reviewed 2026-11-06')).toBeInTheDocument()
    const link = screen.getByRole('link', {
      name: 'New Process for Online Appointment Booking',
    })
    expect(link).toHaveAttribute(
      'href',
      'https://passportindia.gov.in/AppOnlineProject/pdf/New_Online_Appointment_Booking_Process.pdf#page=5'
    )
  })

  it('flow (b): handles clarification via both button click and typed "reissue" reply, plus local "okay thank you" without API call', async () => {
    const clarifyPayload: AskResponse = {
      status: 'clarification',
      answer:
        'Document requirements depend on whether you are applying for a Fresh passport or a Reissue.',
      mode: 'hybrid',
      grounded: false,
      citations: [],
      official_links: [
        {
          label: 'Fresh Passport Document Advisor',
          url: 'https://www.passportindia.gov.in/psp/docAdvisor/attachmentAdvFreshInp',
        },
        {
          label: 'Reissue Document Advisor',
          url: 'https://services2.passportindia.gov.in/psp/docAdvisor/reissuePassport',
        },
      ],
      clarification_question:
        'Is this for a Fresh application (you have never held an ordinary passport) or a Reissue (you previously held one)?',
      llm_used: false,
      error: null,
    }

    const redirectPayload: AskResponse = {
      status: 'official_redirect',
      answer:
        'Because the exact document checklist depends on your reissue reason, use the official Reissue Document Advisor below.',
      mode: 'hybrid',
      grounded: false,
      citations: [],
      official_links: [
        {
          label: 'Reissue Document Advisor',
          url: 'https://services2.passportindia.gov.in/psp/docAdvisor/reissuePassport',
        },
      ],
      clarification_question: null,
      llm_used: false,
      error: null,
    }

    const fetchSpy = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(new Response(JSON.stringify(HEALTH_OK), { status: 200 }))
      .mockResolvedValueOnce(
        new Response(JSON.stringify(clarifyPayload), { status: 200 })
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify(redirectPayload), { status: 200 })
      )

    const user = userEvent.setup()
    render(<App />)

    await user.type(
      screen.getByLabelText('Your question'),
      'What exact documents do I need?'
    )
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await screen.findByRole('button', {
      name: /Reissue \(previously held an ordinary passport\)/i,
    })

    // Type "reissue" instead of clicking the button to verify typed follow-up works identically
    await user.type(screen.getByLabelText('Your question'), 'reissue')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await waitFor(() => expect(fetchSpy).toHaveBeenCalledTimes(3))
    const secondAskBody = JSON.parse(
      String(fetchSpy.mock.calls[2][1]?.body)
    ) as { question: string }
    expect(secondAskBody.question).toBe(
      'For a Normal adult Reissue passport application: What exact documents do I need?'
    )

    // Now send standalone "okay thank you" and confirm NO new fetch call is made
    await user.type(screen.getByLabelText('Your question'), 'okay thank you')
    await user.click(screen.getByRole('button', { name: 'Send' }))
    await screen.findByText(/You're welcome!/i)
    expect(fetchSpy).toHaveBeenCalledTimes(3)
  })

  it('flows (c) & (d): displays conflict without numeric limit and omits unverified live-status link', async () => {
    const conflictPayload: AskResponse = {
      status: 'conflicting_sources',
      answer:
        "Official Passport Seva sources disagree on the number of reschedules allowed, so I can't give a definitive number.",
      mode: 'hybrid',
      grounded: false,
      citations: [],
      official_links: [
        {
          label: 'Process to apply for Fresh / Reissue Ordinary Passport',
          url: 'https://passportindia.gov.in/psp/apply',
        },
      ],
      clarification_question: null,
      llm_used: false,
      error: null,
    }

    const liveStatusPayload: AskResponse = {
      status: 'live_status_redirect',
      answer:
        "Live application or dispatch status isn't available here. Use the official status tracker.",
      mode: 'hybrid',
      grounded: false,
      citations: [],
      official_links: [
        {
          label: 'Passport Seva application status tracker',
          url: 'https://passportindia.gov.in/AppOnlineProject/statusTracker/trackStatusInpNew',
        },
      ],
      clarification_question: null,
      llm_used: false,
      error: null,
    }

    vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(new Response(JSON.stringify(HEALTH_OK), { status: 200 }))
      .mockResolvedValueOnce(
        new Response(JSON.stringify(conflictPayload), { status: 200 })
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify(liveStatusPayload), { status: 200 })
      )

    const user = userEvent.setup()
    render(<App />)

    await user.type(
      screen.getByLabelText('Your question'),
      'How many times can I reschedule?'
    )
    await user.click(screen.getByRole('button', { name: 'Send' }))
    await screen.findByText(/Official Passport Seva sources disagree/i)

    await user.type(
      screen.getByLabelText('Your question'),
      'Has my passport been dispatched today?'
    )
    await user.click(screen.getByRole('button', { name: 'Send' }))
    await screen.findByText(/Direct link omitted because the status-tracker URL/i)
    expect(
      screen.queryByRole('link', {
        name: /Passport Seva application status tracker/i,
      })
    ).toBeNull()
  })

  it('flow (e): shows connection error when backend is unreachable, preserves question, and retries on reconnection', async () => {
    const recoveredPayload: AskResponse = {
      status: 'out_of_scope',
      answer: 'This guide only covers adults applying in India under the Normal process.',
      mode: 'hybrid',
      grounded: false,
      citations: [],
      official_links: [],
      clarification_question: null,
      llm_used: false,
      error: null,
    }

    vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(new Response(JSON.stringify(HEALTH_OK), { status: 200 }))
      .mockRejectedValueOnce(new TypeError('Failed to fetch'))
      .mockResolvedValueOnce(
        new Response(JSON.stringify(recoveredPayload), { status: 200 })
      )
      .mockResolvedValueOnce(new Response(JSON.stringify(HEALTH_OK), { status: 200 }))

    const user = userEvent.setup()
    render(<App />)

    const textarea = screen.getByLabelText('Your question') as HTMLTextAreaElement
    await user.type(textarea, 'How do I apply for Tatkaal for a child?')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await screen.findByText('Backend connection failed')
    // User's question is preserved in the textarea on failure
    expect(textarea.value).toBe('How do I apply for Tatkaal for a child?')

    await user.click(screen.getByRole('button', { name: 'Retry question' }))
    await screen.findByText(/This guide only covers adults/i)
  })
})
