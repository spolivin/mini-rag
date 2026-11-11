// Thin client for the FastAPI backend. Requests go to /api/..., which the Vite dev server
// (and later nginx) forwards to the backend.

export interface StoredDocument {
  doc_id: string
  filename: string
  chunk_size: number
  overlap: number
  created_at: string
  num_chunks: number
}

export interface UploadResult {
  doc_id: string
  filename: string
  created: boolean
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, init)
  if (!response.ok) {
    // FastAPI reports errors as {"detail": "..."}
    let message = `HTTP ${response.status}`
    try {
      const body = await response.json()
      if (typeof body.detail === 'string') message = body.detail
    } catch {
      // the body was not JSON: keep the status line
    }
    throw new Error(message)
  }
  return response.status === 204 ? (undefined as T) : response.json()
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

export function listDocuments(): Promise<StoredDocument[]> {
  return request('/documents')
}

export function uploadDocument(file: File): Promise<UploadResult> {
  const form = new FormData()
  form.append('file', file)
  return request('/documents', { method: 'POST', body: form })
}

export function deleteDocument(docId: string): Promise<void> {
  return request(`/documents/${encodeURIComponent(docId)}`, { method: 'DELETE' })
}

export interface ChunkResult {
  text: string
  score: number
  doc_id: string
  filename: string
}

export interface QueryResult {
  answer: string
  chunks: ChunkResult[]
}

/** Asks a question. An empty `documentIds` searches all documents. */
export function askQuestion(query: string, documentIds: string[]): Promise<QueryResult> {
  return request('/query', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      query,
      document_ids: documentIds.length > 0 ? documentIds : null,
    }),
  })
}
