import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react'
import { askQuestion, errorMessage, type ChunkResult, type StoredDocument } from '../api'

interface Props {
  documents: StoredDocument[]
  /** IDs of the documents the chat is restricted to (empty = all documents). */
  selectedIds: string[]
}

type Message =
  | { role: 'user'; text: string }
  | { role: 'assistant'; text: string; chunks: ChunkResult[] }
  | { role: 'error'; text: string }

function Sources({ chunks }: { chunks: ChunkResult[] }) {
  // Score bars are relative to the best chunk of this answer
  const maxScore = Math.max(0, ...chunks.map((chunk) => chunk.score))
  return (
    <details className="sources">
      <summary>
        {chunks.length} {chunks.length === 1 ? 'source' : 'sources'}
      </summary>
      {chunks.map((chunk, index) => (
        <article className="chunk" key={`${chunk.doc_id}-${index}`}>
          <header>
            <span className="rank">{index + 1}</span>
            <strong>{chunk.filename}</strong>
            <span className="score">score {chunk.score.toFixed(2)}</span>
          </header>
          <div className="bar">
            <span
              style={{
                width: `${maxScore > 0 ? (Math.max(chunk.score, 0) / maxScore) * 100 : 0}%`,
              }}
            />
          </div>
          <p>{chunk.text}</p>
        </article>
      ))}
    </details>
  )
}

export default function Chat({ documents, selectedIds }: Props) {
  const [messages, setMessages] = useState<Message[]>([])
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const threadEnd = useRef<HTMLDivElement>(null)

  useEffect(() => {
    threadEnd.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, loading])

  const scope =
    selectedIds.length === 0
      ? 'all documents'
      : documents
          .filter((doc) => selectedIds.includes(doc.doc_id))
          .map((doc) => doc.filename)
          .join(', ')

  async function send() {
    const query = question.trim()
    if (!query || loading) return

    setMessages((current) => [...current, { role: 'user', text: query }])
    setQuestion('')
    setLoading(true)
    try {
      const result = await askQuestion(query, selectedIds)
      setMessages((current) => [
        ...current,
        { role: 'assistant', text: result.answer, chunks: result.chunks },
      ])
    } catch (e) {
      setMessages((current) => [...current, { role: 'error', text: errorMessage(e) }])
    } finally {
      setLoading(false)
    }
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault()
    void send()
  }

  // Enter sends, Shift+Enter inserts a new line
  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault()
      void send()
    }
  }

  const noDocuments = documents.length === 0

  return (
    <section className="panel chat">
      <header className="panel-header">
        <h2>Chat</h2>
        <span className="scope" title={scope}>
          Searching in: {scope}
        </span>
        {messages.length > 0 && (
          <button type="button" className="link" onClick={() => setMessages([])}>
            Clear
          </button>
        )}
      </header>

      <div className="thread" aria-live="polite">
        {messages.length === 0 && !loading && (
          <p className="empty muted">
            {noDocuments
              ? 'Upload a document on the left to start chatting.'
              : 'Ask a question about your documents. Tick documents on the left to narrow the search.'}
          </p>
        )}

        {messages.map((message, index) => (
          <div className={`message ${message.role}`} key={index}>
            {message.role === 'error' ? (
              <p role="alert">{message.text}</p>
            ) : (
              <p>{message.text}</p>
            )}
            {message.role === 'assistant' && message.chunks.length > 0 && (
              <Sources chunks={message.chunks} />
            )}
          </div>
        ))}

        {loading && (
          <div className="message assistant">
            <p className="muted">
              <span className="spinner" aria-hidden="true" /> Thinking…
            </p>
          </div>
        )}
        <div ref={threadEnd} />
      </div>

      <form className="composer" onSubmit={handleSubmit}>
        <textarea
          rows={2}
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={noDocuments ? 'Upload a document first' : 'Ask a question…'}
          disabled={noDocuments}
        />
        <button type="submit" disabled={loading || noDocuments || !question.trim()}>
          Send
        </button>
      </form>
    </section>
  )
}
