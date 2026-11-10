import { useEffect, useState } from 'react'
import { errorMessage, listDocuments, type StoredDocument } from './api'
import Ask from './components/Ask'
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

  return (
    <main>
      <header className="app-header">
        <div>
          <h1>MiniRAG</h1>
          <p className="tagline">Ask questions about your documents</p>
        </div>
        <p className={`status ${status}`}>
          <span className="dot" aria-hidden="true" />
          Backend {STATUS_TEXT[status]}
        </p>
      </header>
      <Ask documents={documents ?? []} />
      <Documents
        documents={documents}
        loadError={loadError}
        onChange={refreshDocuments}
      />
    </main>
  )
}

export default App
