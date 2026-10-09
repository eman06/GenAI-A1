import { useEffect, useRef, useState } from 'react'
import { PRETTY, download, getJSON, sampleUrl } from '../api.js'

export function ImagePanel({ title, src, caption, onDownload, empty = 'No image yet' }) {
  return (
    <div className="card flex flex-col gap-3 p-4">
      <div className="flex items-center justify-between">
        <span className="label">{title}</span>
        {src && onDownload && (
          <button className="text-xs font-semibold text-accent-400 hover:text-accent-500" onClick={onDownload}>
            Download
          </button>
        )}
      </div>
      <div className="flex aspect-square items-center justify-center overflow-hidden rounded-xl bg-ink-950 ring-1 ring-white/5">
        {src ? (
          <img src={src} alt={title} className="pixelated h-full w-full object-contain" />
        ) : (
          <span className="px-4 text-center text-sm text-slate-500">{empty}</span>
        )}
      </div>
      {caption && <p className="text-xs text-slate-400">{caption}</p>}
    </div>
  )
}

export function Stat({ label, value, hint }) {
  return (
    <div className="rounded-xl bg-ink-800/70 p-3 ring-1 ring-white/5">
      <div className="label">{label}</div>
      <div className="mt-1 text-xl font-semibold text-white">{value}</div>
      {hint && <div className="text-xs text-slate-400">{hint}</div>}
    </div>
  )
}

export function Bars({ title, values, highlight, truth }) {
  const entries = Object.entries(values || {})
  return (
    <div className="card">
      <div className="label mb-3">{title}</div>
      <div className="flex flex-col gap-2.5">
        {entries.map(([k, v]) => (
          <div key={k}>
            <div className="mb-1 flex justify-between text-sm">
              <span className={k === highlight ? 'font-semibold text-accent-400' : 'text-slate-300'}>
                {PRETTY[k] || k}
                {truth && (k === truth || (k === 'identity' && truth === 'clean')) && (
                  <span className="ml-2 rounded bg-emerald-500/15 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-300">TRUE</span>
                )}
              </span>
              <span className="tabular-nums text-slate-300">{(v * 100).toFixed(1)}%</span>
            </div>
            <div className="h-2.5 overflow-hidden rounded-full bg-ink-950">
              <div
                className={`h-full rounded-full transition-all duration-500 ${k === highlight ? 'bg-accent-500' : 'bg-slate-500/60'}`}
                style={{ width: `${Math.max(1, v * 100)}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

/** Upload a file, pick a bundled sample, or (optionally) capture from the webcam. */
export function ImageSource({ value, onChange, allowWebcam = false, samplePrefix = '' }) {
  const [samples, setSamples] = useState([])
  const [cam, setCam] = useState(false)
  const videoRef = useRef(null)
  const streamRef = useRef(null)

  useEffect(() => {
    getJSON('/api/samples').then((r) => setSamples(r.samples.filter((n) => n.startsWith(samplePrefix)))).catch(() => {})
    return () => stopCam()
  }, [])

  function stopCam() {
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    setCam(false)
  }

  async function startCam() {
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
      streamRef.current = s
      setCam(true)
      setTimeout(() => { if (videoRef.current) videoRef.current.srcObject = s }, 0)
    } catch (e) {
      alert('Could not open the webcam: ' + e.message)
    }
  }

  function capture() {
    const v = videoRef.current
    const side = Math.min(v.videoWidth, v.videoHeight)
    const c = document.createElement('canvas')
    c.width = c.height = side
    c.getContext('2d').drawImage(v, (v.videoWidth - side) / 2, (v.videoHeight - side) / 2, side, side, 0, 0, side, side)
    c.toBlob((blob) => {
      const file = new File([blob], 'webcam.png', { type: 'image/png' })
      onChange({ file, preview: URL.createObjectURL(blob), label: 'webcam capture' })
      stopCam()
    }, 'image/png')
  }

  return (
    <div className="card flex flex-col gap-4">
      <div className="label">1 · Input image</div>
      <label className="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-white/20 bg-ink-950/60 p-5 text-center hover:border-accent-500">
        <input
          type="file"
          accept="image/png,image/jpeg,image/webp,image/bmp"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) onChange({ file: f, preview: URL.createObjectURL(f), label: f.name })
          }}
        />
        <span className="text-sm font-semibold text-slate-200">Click to upload</span>
        <span className="text-xs text-slate-500">PNG / JPEG / WebP, max 10 MB, resized to 128×128</span>
      </label>
      {allowWebcam && (
        <div className="flex flex-col gap-2">
          {!cam ? (
            <button className="btn-ghost" onClick={startCam}>Use webcam</button>
          ) : (
            <>
              <video ref={videoRef} autoPlay playsInline muted className="aspect-video w-full rounded-xl bg-black object-cover" />
              <div className="flex gap-2">
                <button className="btn-primary flex-1" onClick={capture}>Capture</button>
                <button className="btn-ghost" onClick={stopCam}>Cancel</button>
              </div>
            </>
          )}
        </div>
      )}
      {samples.length > 0 && (
        <div>
          <div className="mb-2 text-xs text-slate-400">or pick a sample</div>
          <div className="grid grid-cols-4 gap-2">
            {samples.map((s) => (
              <button
                key={s}
                onClick={() => onChange({ sample: s, preview: sampleUrl(s), label: s })}
                className={`overflow-hidden rounded-lg ring-2 ${value?.sample === s ? 'ring-accent-500' : 'ring-transparent hover:ring-white/30'}`}
              >
                <img src={sampleUrl(s)} alt={s} className="aspect-square w-full object-cover" />
              </button>
            ))}
          </div>
        </div>
      )}
      {value?.preview && (
        <div className="flex items-center gap-3 rounded-xl bg-ink-800/60 p-2">
          <img src={value.preview} alt="selected" className="h-12 w-12 rounded-lg object-cover" />
          <span className="truncate text-xs text-slate-300">{value.label}</span>
        </div>
      )}
    </div>
  )
}

export function ErrorBox({ error }) {
  if (!error) return null
  return <div className="rounded-xl border border-rose-500/40 bg-rose-500/10 p-3 text-sm text-rose-200">{error}</div>
}

export function dl(dataUrl, name) {
  return () => download(dataUrl, name)
}
