/**
 * In-memory conversation context tracker and follow-up planner.
 * Keeps the backend API (/api/ask) stateless by expanding contextual follow-ups
 * into complete, self-contained questions while handling standalone small-talk locally.
 */

export type AppType = 'fresh' | 'reissue'
export type Topic = 'application' | 'documents' | 'appointment'

export interface PendingClarification {
  originalQuestion: string
  kind: 'app_type' | 'link_choice'
}

export interface ConversationContext {
  appType: AppType | null
  topic: Topic | null
  pendingClarification: PendingClarification | null
}

export type PlannedAction =
  | {
      kind: 'local'
      displayQuestion: string
      answer: string
      next: ConversationContext
    }
  | {
      kind: 'ask'
      displayQuestion: string
      question: string
      next: ConversationContext
    }

export function initialContext(): ConversationContext {
  return {
    appType: null,
    topic: null,
    pendingClarification: null,
  }
}

const REISSUE_RE =
  /\b(re-?issue\w*|renew\w*|already have (an? )?(ordinary )?passport|previously held|old passport|expired|expiring|about to expire)\b/i
const FRESH_RE =
  /\b(fresh|first(-|\s+)time|first (ordinary )?passport|never held|new passport)\b/i

const DOC_RE =
  /\b(documents?|checklists?|proofs?|papers?|certificates?|advisor)\b/i
const APPOINTMENT_RE =
  /\b(appointments?|book\w*|reschedul\w*|slots?|psk|popsk|move (my|the|it)|shift (my|the)|booked day|visit)\b/i
const APPLICATION_RE =
  /\b(apply\w*|applications?|renew\w*|re-?issue\w*|fresh|register\w*|arn|forms?|submit\w*|start)\b/i

const LINK_FOLLOWUP_RE =
  /\b(which|what)\b[^.?]{0,30}\b(links?|urls?|pages?|websites?|portals?|advisors?)\b|\b(links?|urls?)\b[^.?]{0,25}\b(use|click|open|go to|need)\b|\bwhere (do i|should i|to) (click|go|apply|open)\b/i

/**
 * Matches standalone acknowledgements / thanks / greetings when the message
 * does not also contain a real question.
 */
const ACK_ONLY_RE =
  /^(?:ok(?:ay)?|alright|got it|understood|sure|great|cool|fine|thanks?(?:\s+you)?|thank\s+you(?:\s+so\s+much|\s+very\s+much)?|ty|appreciate it|that helps|helpful|makes sense|perfect|awesome|good to know)[\s,!.;:-]*(?:thanks?(?:\s+you)?|thank\s+you(?:\s+so\s+much|\s+very\s+much)?|ok(?:ay)?|got it|alright|have a good day|bye)?[\s!.]*$/i

const QUESTION_CUE_RE =
  /\?|\b(how|what|which|where|when|why|who|can i|should i|do i|is there|are there|need|require\w*|documents?|checklists?|appointments?|reschedul\w*|re-?issue\w*|renew\w*|fresh|links?|urls?|status|dispatch\w*|psk|arn|fees?|paid|payment|originals?|copies)\b/i

const PURE_APP_TYPE_REPLY_RE =
  /^(?:it(?:'s| is)\s+|for\s+|a\s+|an\s+|normal\s+|adult\s+)*(fresh(?:\s+application|\s+passport)?|re-?issue(?:\s+application|\s+passport)?|renewal|first(?:\s+time|\s+passport)?)[\s.!]*$/i

export function detectAppType(text: string): AppType | null {
  const hasReissue = REISSUE_RE.test(text)
  const hasFresh = FRESH_RE.test(text)
  if (hasReissue && !hasFresh) return 'reissue'
  if (hasFresh && !hasReissue) return 'fresh'
  // If user says "Do I choose Fresh?" while stating "already have an ordinary passport"
  if (/\balready have\b|\bpreviously held\b|\bold passport\b|\bexpired\b/i.test(text)) {
    return 'reissue'
  }
  if (/\bfirst(-|\s+)time\b|\bfirst (ordinary )?passport\b|\bnever held\b/i.test(text)) {
    return 'fresh'
  }
  return null
}

export function detectTopic(text: string): Topic | null {
  if (DOC_RE.test(text)) return 'documents'
  if (APPOINTMENT_RE.test(text)) return 'appointment'
  if (APPLICATION_RE.test(text)) return 'application'
  return null
}

export function isStandaloneAcknowledgement(text: string): boolean {
  const trimmed = text.trim()
  if (!trimmed) return false
  if (QUESTION_CUE_RE.test(trimmed)) return false
  return ACK_ONLY_RE.test(trimmed)
}

function stripLeadingAck(text: string): string {
  return text
    .replace(
      /^(?:(?:ok(?:ay)?|alright|got it|sure|great|thanks?(?:\s+you)?|thank\s+you(?:\s+so\s+much)?)[\s,!.;:-]+)+/i,
      ''
    )
    .trim()
}

function parsePureAppTypeReply(text: string): AppType | null {
  const m = text.trim().match(PURE_APP_TYPE_REPLY_RE)
  if (!m) return null
  const token = m[1].toLowerCase()
  if (token.startsWith('re') || token.startsWith('renew')) return 'reissue'
  return 'fresh'
}

function formatAppTypeLabel(appType: AppType): 'Fresh' | 'Reissue' {
  return appType === 'reissue' ? 'Reissue' : 'Fresh'
}

/**
 * Plan how to handle a user message (either typed in the composer or triggered
 * via a clarification choice button) given the current conversation context.
 */
export function planMessage(
  rawInput: string,
  ctx: ConversationContext
): PlannedAction {
  const displayQuestion = rawInput.trim()

  // 1. Standalone acknowledgement ("okay thank you", "thanks!", etc.):
  //    Respond locally without calling the API and preserve all context.
  if (isStandaloneAcknowledgement(displayQuestion)) {
    return {
      kind: 'local',
      displayQuestion,
      answer:
        "You're welcome! Let me know if you have any other questions about your passport application, required documents, or appointments.",
      next: { ...ctx },
    }
  }

  // Strip a leading "Okay thank you, ..." prefix when followed by a real question
  const coreQuestion = stripLeadingAck(displayQuestion) || displayQuestion

  // 2. If there is a pending clarification (e.g., assistant asked Fresh vs Reissue
  //    after "What exact documents do I need?") and the user replies with "reissue"
  //    or clicks the Reissue/Fresh button:
  const replyType = parsePureAppTypeReply(coreQuestion)
  if (replyType && ctx.pendingClarification?.kind === 'app_type') {
    const label = formatAppTypeLabel(replyType)
    const orig = ctx.pendingClarification.originalQuestion
    return {
      kind: 'ask',
      displayQuestion,
      question: `For a Normal adult ${label} passport application: ${orig}`,
      next: {
        appType: replyType,
        topic: 'documents',
        pendingClarification: null,
      },
    }
  }

  // If the user typed just "reissue" or "fresh" without a pending clarification,
  // treat it according to the active topic (or general application guidance).
  if (replyType && !ctx.pendingClarification) {
    const label = formatAppTypeLabel(replyType)
    const question =
      ctx.topic === 'documents'
        ? `For a Normal adult ${label} passport application: What exact documents do I need?`
        : `How do I apply for a Normal adult ${label} ordinary passport?`
    return {
      kind: 'ask',
      displayQuestion,
      question,
      next: {
        appType: replyType,
        topic: ctx.topic ?? 'application',
        pendingClarification: null,
      },
    }
  }

  // 3. Determine updated application type and topic from the new question.
  //    An explicit app type in the question overrides the remembered one.
  const explicitAppType = detectAppType(coreQuestion)
  const explicitTopic = detectTopic(coreQuestion)

  // If the user switches to an appointment procedural question (like booking or moving
  // a slot), update topic to 'appointment'. Keep appType unless overridden.
  const effectiveAppType = explicitAppType ?? ctx.appType
  const effectiveTopic = explicitTopic ?? ctx.topic

  // 4. Handle "Which link should I use?" / "Where do I click?" in context:
  if (LINK_FOLLOWUP_RE.test(coreQuestion)) {
    if (explicitTopic === 'documents' || (!explicitTopic && ctx.topic === 'documents')) {
      if (effectiveAppType) {
        const label = formatAppTypeLabel(effectiveAppType)
        return {
          kind: 'ask',
          displayQuestion,
          question: `Which official Document Advisor link should I use for a ${label} passport?`,
          next: {
            appType: effectiveAppType,
            topic: 'documents',
            pendingClarification: null,
          },
        }
      }
      return {
        kind: 'ask',
        displayQuestion,
        question: 'Which official Document Advisor link should I use?',
        next: {
          appType: null,
          topic: 'documents',
          pendingClarification: {
            originalQuestion: 'What exact documents do I need?',
            kind: 'app_type',
          },
        },
      }
    }

    if (explicitTopic === 'appointment' || (!explicitTopic && ctx.topic === 'appointment')) {
      return {
        kind: 'ask',
        displayQuestion,
        question:
          'Which official link should I use for booking or rescheduling a passport appointment?',
        next: {
          appType: effectiveAppType,
          topic: 'appointment',
          pendingClarification: null,
        },
      }
    }

    if (effectiveAppType) {
      const label = formatAppTypeLabel(effectiveAppType)
      return {
        kind: 'ask',
        displayQuestion,
        question: `Which official link should I use to apply for a ${label} ordinary passport?`,
        next: {
          appType: effectiveAppType,
          topic: 'application',
          pendingClarification: null,
        },
      }
    }

    // No prior context to disambiguate which link the user wants:
    return {
      kind: 'ask',
      displayQuestion,
      question: coreQuestion,
      next: {
        appType: effectiveAppType,
        topic: effectiveTopic,
        pendingClarification: {
          originalQuestion: coreQuestion,
          kind: 'link_choice',
        },
      },
    }
  }

  // 5. Document questions ("What documents do I need?", "What exact documents do I need?"):
  //    - If the question already states an explicit app type ("What documents do I need for a first time passport?"),
  //      send the question as-is (or with explicit type) and update appType.
  //    - If the question does not state an app type, but effectiveAppType is already known
  //      from earlier in the chat (e.g., after a reissue question), attach the known type
  //      so the assistant does NOT ask Fresh vs Reissue again!
  if (explicitTopic === 'documents') {
    const bringOriginalsOnly =
      /\b(originals?|photocopies|copies|self-attested)\b/i.test(coreQuestion) &&
      !/\b(what|which|list|checklist|exact|complete|full)\b/i.test(coreQuestion)

    if (!bringOriginalsOnly) {
      if (explicitAppType) {
        return {
          kind: 'ask',
          displayQuestion,
          question: coreQuestion,
          next: {
            appType: explicitAppType,
            topic: 'documents',
            pendingClarification: null,
          },
        }
      }
      if (effectiveAppType) {
        const label = formatAppTypeLabel(effectiveAppType)
        return {
          kind: 'ask',
          displayQuestion,
          question: `For a Normal adult ${label} passport application: ${coreQuestion}`,
          next: {
            appType: effectiveAppType,
            topic: 'documents',
            pendingClarification: null,
          },
        }
      }
      // Neither explicit nor remembered appType -> backend will return clarification
      return {
        kind: 'ask',
        displayQuestion,
        question: coreQuestion,
        next: {
          appType: null,
          topic: 'documents',
          pendingClarification: {
            originalQuestion: coreQuestion,
            kind: 'app_type',
          },
        },
      }
    }
  }

  // 6. Standard question (or topic change, e.g. from reissue documents to booking/moving an appointment):
  //    Send the question directly without polluting unrelated topics with document prefixes.
  return {
    kind: 'ask',
    displayQuestion,
    question: coreQuestion,
    next: {
      appType: effectiveAppType,
      topic: effectiveTopic,
      pendingClarification: null,
    },
  }
}
