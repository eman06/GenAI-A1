const BASE = import.meta.env.VITE_API_BASE || ''

async function handle(res) {
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(body.detail || `HTTP ${res.status}`)
  return body
}

export const getJSON = (path) => fetch(`${BASE}${path}`).then(handle)

export function postForm(path, fields) {
  const fd = new FormData()
  Object.entries(fields).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== '') fd.append(k, v)
  })
  return fetch(`${BASE}${path}`, { method: 'POST', body: fd }).then(handle)
}

export const sampleUrl = (name) => `${BASE}/api/samples/${encodeURIComponent(name)}`

export function download(dataUrl, filename) {
  const a = document.createElement('a')
  a.href = dataUrl
  a.download = filename
  a.click()
}

export const PRETTY = {
  clean: 'Clean', salt_pepper: 'Salt & pepper', blur: 'Gaussian blur', occlusion: 'Occlusion',
  identity: 'Identity', none: 'None (upload already corrupted)',
}
