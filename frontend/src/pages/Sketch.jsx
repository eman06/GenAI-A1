import { useState } from 'react'
import { postForm } from '../api.js'
import { ErrorBox, ImagePanel, ImageSource, Stat, dl } from '../components/ui.jsx'

const STYLES = ['Style 1', 'Style 2', 'Style 3']

export default function Sketch() {
  const [src, setSrc] = useState(null)
  const [style, setStyle] = useState(0)
  const [res, setRes] = useState(null)
  const [all, setAll] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  async function run() {
    setBusy(true)
    setError('')
    try {
      setRes(await postForm('/api/sketch', { file: src?.file, sample: src?.sample, style }))
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  async function runAll() {
    setBusy(true)
    setError('')
    try {
      const out = []
      for (let s = 0; s < 3; s++) out.push(await postForm('/api/sketch', { file: src?.file, sample: src?.sample, style: s }))
      setAll(out)
      setRes(out[style])
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <header>
        <h1 className="text-2xl font-bold text-white">Face-to-Sketch Generator</h1>
        <p className="mt-1 max-w-3xl text-sm text-slate-400">
          A U-Net generator conditioned on a learned style embedding (Task 4, trained adversarially against a
          PatchGAN discriminator on FS2K) turns a face photograph into a sketch in one of three styles.
        </p>
      </header>
      <div className="grid gap-6 xl:grid-cols-[320px_1fr]">
        <aside className="flex flex-col gap-4">
          <ImageSource value={src} onChange={(v) => { setSrc(v); setAll(null) }} allowWebcam samplePrefix="face" />
          <div className="card flex flex-col gap-3">
            <div className="label">2 · Sketch style</div>
            <div className="grid grid-cols-3 gap-2">
              {STYLES.map((s, i) => (
                <button key={s} onClick={() => setStyle(i)}
                  className={`rounded-lg px-2 py-2.5 text-sm font-semibold ${style === i ? 'bg-accent-500 text-ink-950' : 'bg-ink-800 text-slate-300 hover:bg-ink-700'}`}>
                  {s}
                </button>
              ))}
            </div>
          </div>
          <button className="btn-primary py-3" disabled={!src || busy} onClick={run}>
            {busy ? 'Generating…' : 'Generate sketch'}
          </button>
          <button className="btn-ghost" disabled={!src || busy} onClick={runAll}>
            Compare all 3 styles
          </button>
          <ErrorBox error={error} />
        </aside>
        <section className="flex flex-col gap-6">
          <div className="grid gap-4 sm:grid-cols-2">
            <ImagePanel title="Photograph (128×128 model input)" src={res?.images.photo} empty="Upload, capture or pick a face photo" />
            <ImagePanel title={`Generated sketch${res ? ` · ${res.style}` : ''}`} src={res?.images.sketch}
              onDownload={res && dl(res.images.sketch, `sketch_${res.style.replace(' ', '').toLowerCase()}.png`)} />
          </div>
          {res && (
            <div className="grid gap-4 sm:grid-cols-3">
              <Stat label="Inference time" value={`${res.timing_ms.total.toFixed(1)} ms`} hint="ONNX Runtime, CPU" />
              <Stat label="Style condition" value={res.style} hint="learned categorical embedding" />
              <Stat label="Original size" value={res.input.original_size.join('×')} hint="resized to 128×128" />
            </div>
          )}
          {all && (
            <div className="card">
              <div className="label mb-3">Style comparison: same photo, three style conditions</div>
              <div className="grid grid-cols-3 gap-4">
                {all.map((r) => (
                  <div key={r.style} className="flex flex-col gap-2">
                    <img src={r.images.sketch} alt={r.style} className="pixelated aspect-square w-full rounded-xl bg-white" />
                    <div className="flex items-center justify-between text-sm text-slate-300">
                      {r.style}
                      <button className="text-xs font-semibold text-accent-400" onClick={dl(r.images.sketch, `sketch_${r.style.replace(' ', '').toLowerCase()}.png`)}>Download</button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
