/**
 * Screening assistant: a side sheet where staff can ask an LLM agent about the current visit. The
 * agent reads the visit through server-side tools and never changes a grade.
 */
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { api } from '../api'
import { buildVisitContext, SUGGESTED_PROMPTS, TOOL_LABELS } from '../assistant'
import type { EyeCase, EyeSide } from '../types'
import { CloseIcon } from './Icons'

const MAX_HISTORY = 20

interface Turn {
  role: 'user' | 'assistant'
  content: string
  tools?: string[]
}

interface AssistantPanelProps {
  open: boolean
  onClose: () => void
  eyes: Record<EyeSide, EyeCase | null>
}

export function AssistantPanel({ open, onClose, eyes }: AssistantPanelProps) {
  const [turns, setTurns] = useState<Turn[]>([])
  const [input, setInput] = useState('')
  const [waiting, setWaiting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [copied, setCopied] = useState<number | null>(null)
  const panelRef = useRef<HTMLElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const endRef = useRef<HTMLDivElement>(null)

  const canChat = Object.values(eyes).some((e) => e?.status === 'done' && e.result)

  useEffect(() => {
    if (open) inputRef.current?.focus()
  }, [open])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [turns, waiting, error])

  useEffect(() => {
    const box = inputRef.current
    if (!box) return
    box.style.height = 'auto'
    box.style.height = `${box.scrollHeight}px`
  }, [input])

  /** Send the conversation (ending with a user turn) and append the reply. */
  const ask = async (history: Turn[]) => {
    setTurns(history)
    setError(null)
    setWaiting(true)
    try {
      const reply = await api.assistant(
        history.slice(-MAX_HISTORY).map(({ role, content }) => ({ role, content })),
        buildVisitContext(eyes),
      )
      setTurns([...history, { role: 'assistant', content: reply.reply, tools: reply.tools_used }])
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setWaiting(false)
    }
  }

  const send = (text: string) => {
    const content = text.trim()
    if (!content || waiting || !canChat) return
    setInput('')
    void ask([...turns, { role: 'user', content }])
  }

  const copy = (index: number, text: string) => {
    navigator.clipboard
      .writeText(text)
      .then(() => {
        setCopied(index)
        setTimeout(() => setCopied((c) => (c === index ? null : c)), 2000)
      })
      .catch(() => undefined)
  }

  if (!open) return null

  return (
    <aside
      ref={panelRef}
      className="assistant-panel"
      aria-label="Screening assistant"
      onKeyDown={(e) => {
        if (e.key === 'Escape') {
          e.stopPropagation()
          onClose()
        }
      }}
    >
      <div className="assistant-header">
        <div>
          <h2 className="assistant-title">Screening assistant</h2>
          <p className="assistant-caption">AI agent · Powered by Groq</p>
        </div>
        <button type="button" className="icon-btn" onClick={onClose} title="Close (Esc)">
          <CloseIcon />
          <span className="visually-hidden">Close assistant</span>
        </button>
      </div>

      <p className="assistant-disclaimer">
        Decision support only. It explains the model's output and never changes a grade.
      </p>

      <div className="assistant-messages">
        {turns.length === 0 ? (
          <div className="assistant-empty">
            <p className="assistant-intro">
              Ask about this visit. The assistant looks up the grades, probabilities, image quality and
              heatmap focus, and checks them against the ICDR grading guidelines.
            </p>
            <div className="assistant-chips">
              {SUGGESTED_PROMPTS.map((prompt) => (
                <button
                  key={prompt}
                  type="button"
                  className="chip"
                  onClick={() => send(prompt)}
                  disabled={waiting || !canChat}
                >
                  {prompt}
                </button>
              ))}
            </div>
            {!canChat && <p className="assistant-hint">Grade a photograph first.</p>}
          </div>
        ) : (
          turns.map((turn, index) => (
            <div key={index} className={`message message-${turn.role}`}>
              <div className="message-content">
                {turn.role === 'assistant' ? <Markdown text={turn.content} /> : <p className="message-p">{turn.content}</p>}
              </div>
              {turn.role === 'assistant' && (
                <div className="message-meta">
                  {turn.tools && turn.tools.length > 0 && (
                    <span className="message-tools">
                      Used: {turn.tools.map((t) => TOOL_LABELS[t] ?? t).join(' · ')}
                    </span>
                  )}
                  <button type="button" className="message-copy" onClick={() => copy(index, turn.content)}>
                    {copied === index ? 'Copied' : 'Copy'}
                  </button>
                </div>
              )}
            </div>
          ))
        )}
        {waiting && (
          <div className="message message-assistant" aria-live="polite">
            <div className="message-content typing-indicator">
              <span className="visually-hidden">Assistant is thinking</span>
              <span className="typing-dot" />
              <span className="typing-dot" />
              <span className="typing-dot" />
            </div>
          </div>
        )}
        {error && (
          <div className="assistant-error" role="alert">
            <p>{error}</p>
            {turns.at(-1)?.role === 'user' && (
              <button type="button" className="btn btn-quiet" onClick={() => void ask(turns)}>
                Retry
              </button>
            )}
          </div>
        )}
        <div ref={endRef} />
      </div>

      <form
        className="assistant-composer"
        onSubmit={(e) => {
          e.preventDefault()
          send(input)
        }}
      >
        <label htmlFor="assistant-input" className="visually-hidden">
          Message the screening assistant
        </label>
        <textarea
          id="assistant-input"
          ref={inputRef}
          className="composer-input"
          placeholder={canChat ? 'Ask about this visit…' : 'Grade a photograph first.'}
          value={input}
          onChange={(e) => setInput(e.currentTarget.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault()
              send(input)
            }
          }}
          disabled={!canChat}
          rows={1}
          maxLength={4000}
        />
        <button type="submit" className="btn btn-primary composer-send" disabled={waiting || !canChat || !input.trim()}>
          Send
        </button>
      </form>
    </aside>
  )
}

/** **bold** and `code` inside a line, as React nodes (never as HTML). */
function inline(text: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**') && part.length > 4) return <strong key={i}>{part.slice(2, -2)}</strong>
    if (part.startsWith('`') && part.endsWith('`') && part.length > 2) return <code key={i}>{part.slice(1, -1)}</code>
    return part
  })
}

const BULLET = /^\s*[-*•]\s+/
const NUMBERED = /^\s*\d+[.)]\s+/
const HEADING = /^\s*#{1,6}\s+/
const TABLE_ROW = /^\s*\|.*\|\s*$/
const TABLE_RULE = /^\s*\|?[\s:|-]+\|?\s*$/

/** A small, safe markdown subset: headings, paragraphs, bullet/numbered lists and pipe tables. */
function Markdown({ text }: { text: string }) {
  const lines = text.replace(/\r/g, '').split('\n')
  const blocks: ReactNode[] = []
  let i = 0
  while (i < lines.length) {
    const line = lines[i]
    if (!line.trim()) {
      i++
    } else if (HEADING.test(line)) {
      blocks.push(
        <p key={i} className="message-p message-heading">
          {inline(line.replace(HEADING, '').replace(/\*\*/g, ''))}
        </p>,
      )
      i++
    } else if (BULLET.test(line) || NUMBERED.test(line)) {
      const ordered = NUMBERED.test(line)
      const pattern = ordered ? NUMBERED : BULLET
      const items: string[] = []
      while (i < lines.length && pattern.test(lines[i])) items.push(lines[i++].replace(pattern, ''))
      const List = ordered ? 'ol' : 'ul'
      blocks.push(
        <List key={i} className="message-list">
          {items.map((item, k) => (
            <li key={k}>{inline(item)}</li>
          ))}
        </List>,
      )
    } else if (TABLE_ROW.test(line)) {
      const rows: string[][] = []
      while (i < lines.length && TABLE_ROW.test(lines[i])) {
        if (!TABLE_RULE.test(lines[i])) rows.push(lines[i].trim().slice(1, -1).split('|').map((c) => c.trim()))
        i++
      }
      const [head, ...body] = rows
      blocks.push(
        <table key={i} className="message-table">
          <thead>
            <tr>
              {head.map((c, k) => (
                <th key={k}>{inline(c)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {body.map((row, r) => (
              <tr key={r}>
                {row.map((c, k) => (
                  <td key={k}>{inline(c)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>,
      )
    } else {
      const para: string[] = []
      while (
        i < lines.length &&
        lines[i].trim() &&
        ![HEADING, BULLET, NUMBERED, TABLE_ROW].some((p) => p.test(lines[i]))
      )
        para.push(lines[i++])
      blocks.push(
        <p key={i} className="message-p">
          {para.map((l, k) => (
            <span key={k}>
              {k > 0 && <br />}
              {inline(l)}
            </span>
          ))}
        </p>,
      )
    }
  }
  return <>{blocks}</>
}
