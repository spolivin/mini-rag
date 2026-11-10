import { useState, type ChangeEvent } from 'react'
import { deleteDocument, errorMessage, uploadDocument, type StoredDocument } from '../api'

interface Props {
  documents: StoredDocument[] | null
  loadError: string | null
  /** Called after an upload or delete so the parent reloads the list. */
  onChange: () => Promise<void>
}

export default function Documents({ documents, loadError, onChange }: Props) {
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
    <section>
      <h2>Documents</h2>

      <input
        type="file"
        accept=".pdf,.txt,.md"
        onChange={handleUpload}
        disabled={uploading}
      />
      {uploading && (
        <p className="muted">Uploading and indexing… this can take a few seconds.</p>
      )}
      {notice && <p className="notice">{notice}</p>}
      {shownError && (
        <p className="error" role="alert">
          {shownError}
        </p>
      )}

      {documents === null ? (
        !shownError && <p className="muted">Loading…</p>
      ) : documents.length === 0 ? (
        <p className="muted">No documents yet. Upload a PDF, TXT or Markdown file.</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>File</th>
              <th className="num">Chunks</th>
              <th>Added</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {documents.map((doc) => (
              <tr key={doc.doc_id}>
                <td>{doc.filename}</td>
                <td className="num">{doc.num_chunks}</td>
                <td>{new Date(doc.created_at).toLocaleString()}</td>
                <td>
                  <button
                    type="button"
                    className="danger"
                    onClick={() => handleDelete(doc)}
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  )
}
