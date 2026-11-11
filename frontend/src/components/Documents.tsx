import { useState, type ChangeEvent } from 'react'
import { deleteDocument, errorMessage, uploadDocument, type StoredDocument } from '../api'

interface Props {
  documents: StoredDocument[] | null
  loadError: string | null
  /** IDs of the documents the chat is restricted to. */
  selectedIds: string[]
  onToggle: (docId: string) => void
  /** Called after an upload or delete so the parent reloads the list. */
  onChange: () => Promise<void>
}

export default function Documents({
  documents,
  loadError,
  selectedIds,
  onToggle,
  onChange,
}: Props) {
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const shownError = error ?? loadError

  async function handleUpload(event: ChangeEvent<HTMLInputElement>) {
    const input = event.currentTarget
    const file = input.files?.[0]
    if (!file) return

    setUploading(true)
    setError(null)
    setNotice(null)
    try {
      const result = await uploadDocument(file)
      setNotice(
        result.created
          ? `Added ${result.filename}.`
          : `${result.filename} had already been uploaded.`,
      )
      await onChange()
    } catch (e) {
      setError(errorMessage(e))
    } finally {
      setUploading(false)
      input.value = '' // allow selecting the same file again
    }
  }

  async function handleDelete(doc: StoredDocument) {
    if (!window.confirm(`Delete ${doc.filename}? Its indexed chunks are removed too.`)) {
      return
    }
    setError(null)
    setNotice(null)
    try {
      await deleteDocument(doc.doc_id)
      await onChange()
    } catch (e) {
      setError(errorMessage(e))
    }
  }

  return (
    <aside className="panel documents">
      <header className="panel-header">
        <h2>Documents</h2>
      </header>

      <div className="upload">
        <input
          type="file"
          accept=".pdf,.txt,.md"
          onChange={handleUpload}
          disabled={uploading}
          aria-label="Upload a document"
        />
        {uploading && <p className="muted small">Uploading and indexing…</p>}
        {notice && <p className="notice">{notice}</p>}
        {shownError && (
          <p className="error" role="alert">
            {shownError}
          </p>
        )}
      </div>

      {documents === null ? (
        !shownError && <p className="muted empty">Loading…</p>
      ) : documents.length === 0 ? (
        <p className="muted empty">No documents yet. Upload a PDF, TXT or Markdown file.</p>
      ) : (
        <>
          <p className="muted small hint">
            Tick documents to chat with only those (none ticked = all).
          </p>
          <ul className="doc-list">
            {documents.map((doc) => (
              <li key={doc.doc_id} className={selectedIds.includes(doc.doc_id) ? 'selected' : ''}>
                <label>
                  <input
                    type="checkbox"
                    checked={selectedIds.includes(doc.doc_id)}
                    onChange={() => onToggle(doc.doc_id)}
                  />
                  <span className="doc-info">
                    <span className="doc-name">{doc.filename}</span>
                    <span className="muted small">
                      {doc.num_chunks} chunks · {new Date(doc.created_at).toLocaleDateString()}
                    </span>
                  </span>
                </label>
                <button
                  type="button"
                  className="danger"
                  onClick={() => handleDelete(doc)}
                  aria-label={`Delete ${doc.filename}`}
                >
                  Delete
                </button>
              </li>
            ))}
          </ul>
        </>
      )}
    </aside>
  )
}
