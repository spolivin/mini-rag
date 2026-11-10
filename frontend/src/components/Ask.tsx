import { useState, type FormEvent } from 'react'
import { askQuestion, errorMessage, type QueryResult, type StoredDocument } from '../api'

interface Props {
  documents: StoredDocument[]
}

export default function Ask({ documents }: Props) {
  const [question, setQuestion] = useState('')
  const [selected, setSelected] = useState<string[]>([])
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<QueryResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Ignore selected documents that have been deleted in the meantime
  const documentIds = selected.filter((id) => documents.some((doc) => doc.doc_id === id))

  function toggle(docId: string) {
    setSelected((current) =>
      current.includes(docId) ? current.filter((id) => id !== docId) : [...current, docId],
    )
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault()
    const query = question.trim()
    if (!query) return

    setLoading(true)
    setError(null)
    try {
      setResult(await askQuestion(query, documentIds))
    } catch (e) {
      setResult(null)
      setError(errorMessage(e))
    } finally {
      setLoading(false)
    }
  }

  // Score bars are relative to the best chunk of this answer
  const maxScore = result ? Math.max(0, ...result.chunks.map((chunk) => chunk.score)) : 0

  return (
    <section>
      <h2>Ask</h2>

      {documents.length === 0 ? (
        <p className="muted">Upload a document first.</p>
      ) : (
        <form onSubmit={handleSubmit}>
          <fieldset>
            <legend>
              Search in <span className="muted">(none selected = all documents)</span>
            </legend>
            {documents.map((doc) => (
              <label key={doc.doc_id}>
                <input
                  type="checkbox"
                  checked={documentIds.includes(doc.doc_id)}
                  onChange={() => toggle(doc.doc_id)}
                />{' '}
                {doc.filename}
              </label>
            ))}
          </fieldset>

          <textarea
            rows={3}
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask a question about your documents"
          />
          <button type="submit" disabled={loading || !question.trim()}>
            {loading && <span className="spinner" aria-hidden="true" />}
            {loading ? 'Thinking…' : 'Ask'}
          </button>
        </form>
      )}

      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}

      {result && (
        <>
          <h3>Answer</h3>
          <p className="answer">{result.answer}</p>

          {result.chunks.length > 0 && <h3>Sources</h3>}
          {result.chunks.map((chunk, index) => (
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
        </>
      )}
    </section>
  )
}
