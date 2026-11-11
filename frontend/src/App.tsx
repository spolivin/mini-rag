import { useEffect, useState } from 'react'
import { errorMessage, listDocuments, type StoredDocument } from './api'
import Chat from './components/Chat'
import Documents from './components/Documents'

type Status = 'checking' | 'ok' | 'down'

const STATUS_TEXT: Record<Status, string> = {
  checking: 'checking…',
  ok: 'connected',
  down: 'unreachable',
}

function App() {
  const [status, setStatus] = useState<Status>('checking')

  useEffect(() => {
    const controller = new AbortController()
    fetch('/api/health', { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        return response.json()
      })
      .then((body) => setStatus(body.status === 'ok' ? 'ok' : 'down'))
      .catch((error) => {
        if (error.name !== 'AbortError') setStatus('down')
      })
    return () => controller.abort()
  }, [])

  // The document list is shared by the Ask and Documents sections
  const [documents, setDocuments] = useState<StoredDocument[] | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)

  // Reload after an upload or delete
  async function refreshDocuments() {
    try {
      setDocuments(await listDocuments())
      setLoadError(null)
    } catch (e) {
      setLoadError(errorMessage(e))
    }
  }

  // Initial load
  useEffect(() => {
    let ignore = false
    listDocuments()
      .then((docs) => {
        if (!ignore) setDocuments(docs)
      })
      .catch((e) => {
        if (!ignore) setLoadError(errorMessage(e))
      })
    return () => {
      ignore = true
    }
  }, [])

  // Documents the chat is restricted to (empty = all). Ids of deleted documents are ignored.
  const [selected, setSelected] = useState<string[]>([])
  const selectedIds = selected.filter((id) => documents?.some((doc) => doc.doc_id === id))

  function toggleSelected(docId: string) {
    setSelected((current) =>
      current.includes(docId) ? current.filter((id) => id !== docId) : [...current, docId],
    )
  }

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>MiniRAG</h1>
          <p className="tagline">Chat with your documents</p>
        </div>
        <p className={`status ${status}`}>
          <span className="dot" aria-hidden="true" />
          Backend {STATUS_TEXT[status]}
        </p>
      </header>
      <div className="layout">
        <Documents
          documents={documents}
          loadError={loadError}
          selectedIds={selectedIds}
          onToggle={toggleSelected}
          onChange={refreshDocuments}
        />
        <Chat documents={documents ?? []} selectedIds={selectedIds} />
      </div>
    </div>
  )
}

export default App
