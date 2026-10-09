import { useState } from 'react'
import { PRETTY, postForm } from '../api.js'
import { Bars, ErrorBox, ImagePanel, ImageSource, Stat, dl } from '../components/ui.jsx'

const MODES = {
  universal: {
    endpoint: '/api/restore/universal',
    title: 'Universal Restoration',
    blurb: 'One convolutional denoising autoencoder (Task 1) restores clean, salt-and-pepper, blurred and occluded images without being told which corruption is present.',
  },
  hard: {
    endpoint: '/api/restore/hard',
    title: 'Hard-Routed Restoration',
    blurb: 'A CNN classifier (Task 2) predicts the corruption type and sends the image to exactly one specialist autoencoder. Clean images take an identity bypass.',
  },
  soft: {
    endpoint: '/api/restore/soft',
    title: 'Soft Mixture-of-Experts Restoration',
    blurb: 'A gating network (Task 3) assigns a continuous weight to the identity branch and all three specialists; the output is their weighted sum.',
  },
}

const SETTINGS = {
  salt_pepper: ['p = 0.03', 'p = 0.08', 'p = 0.15'],
  blur: ['k = 3, σ = 0.7', 'k = 5, σ = 1.5', 'k = 7, σ = 2.5'],
  occlusion: ['1 rect ≈ 10%', '2 rects ≈ 20%', '3 rects ≈ 35%'],
}

function describe(spec) {
  if (!spec) return 'none applied: the uploaded image is treated as the corrupted input'
  if (spec.type === 'clean') return 'clean (no corruption)'
  const base = `${PRETTY[spec.type]}, ${spec.severity}`
  if (spec.type === 'salt_pepper') return `${base} · p = ${spec.prob}`
  if (spec.type === 'blur') return `${base} · kernel ${spec.kernel}, σ = ${spec.sigma}`
  return `${base} · ${spec.n_rects} rect(s), ${(spec.coverage * 100).toFixed(1)}% covered`
}

export default function Restoration({ mode }) {
  const cfg = MODES[mode]
  const [src, setSrc] = useState(null)
  const [corruption, setCorruption] = useState('salt_pepper')
  const [level, setLevel] = useState(1)
  const [seed, setSeed] = useState(0)
  const [res, setRes] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  async function run() {
    setBusy(true)
    setError('')
    try {
      const r = await postForm(cfg.endpoint, {
        file: src?.file, sample: src?.sample, corruption, level, seed,
      })
      setRes(r)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const m = res?.metrics
  return (
    <div className="flex flex-col gap-6">
      <header>
        <h1 className="text-2xl font-bold text-white">{cfg.title}</h1>
        <p className="mt-1 max-w-3xl text-sm text-slate-400">{cfg.blurb}</p>
      </header>

      <div className="grid gap-6 xl:grid-cols-[320px_1fr]">
        <aside className="flex flex-col gap-4">
          <ImageSource value={src} onChange={setSrc} samplePrefix="pet" />
          <div className="card flex flex-col gap-3">
            <div className="label">2 · Runtime corruption</div>
            <select className="select" value={corruption} onChange={(e) => setCorruption(e.target.value)}>
              <option value="none">None (upload already corrupted)</option>
              <option value="clean">Clean (no corruption)</option>
              <option value="salt_pepper">Salt & pepper noise</option>
              <option value="blur">Gaussian blur</option>
              <option value="occlusion">Rectangular occlusion</option>
            </select>
            {SETTINGS[corruption] && (
              <>
                <div className="grid grid-cols-3 gap-2">
                  {['Low', 'Medium', 'High'].map((l, i) => (
                    <button key={l} onClick={() => setLevel(i)}
                      className={`rounded-lg px-2 py-2 text-xs font-semibold ${level === i ? 'bg-accent-500 text-ink-950' : 'bg-ink-800 text-slate-300 hover:bg-ink-700'}`}>
                      {l}
                    </button>
                  ))}
                </div>
                <div className="text-xs text-slate-400">{SETTINGS[corruption][level]} (test-set severity)</div>
                <label className="flex items-center justify-between gap-3 text-xs text-slate-400">
                  Random seed
                  <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))}
                    className="select w-24 py-1" />
                </label>
              </>
            )}
          </div>
          <button className="btn-primary py-3" disabled={!src || busy} onClick={run}>
            {busy ? 'Running…' : 'Run model'}
          </button>
          <ErrorBox error={error} />
        </aside>

        <section className="flex flex-col gap-6">
          <div className="grid gap-4 sm:grid-cols-2 2xl:grid-cols-4">
            <ImagePanel title="Clean reference" src={res?.images.clean}
              empty={res ? 'Not available for uploaded corrupted images' : 'Run the model to see results'} />
            <ImagePanel title="Model input" src={res?.images.input} caption={res && describe(res.corruption)}
              onDownload={res && dl(res.images.input, 'input.png')} />
            <ImagePanel title="Restored output" src={res?.images.output}
              onDownload={res && dl(res.images.output, `restored_${mode}.png`)} />
            <ImagePanel title="Absolute error map" src={res?.images.error_map}
              caption={res && (res.images.clean ? '|output − clean|' : '|output − input| (no clean reference)')} />
          </div>

          {res && (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <Stat label="Inference time" value={`${res.timing_ms.total.toFixed(1)} ms`}
                hint={Object.entries(res.timing_ms).filter(([k]) => k !== 'total').map(([k, v]) => `${k} ${v.toFixed(1)} ms`).join(' · ')} />
              {m?.input_vs_clean && <Stat label="Input PSNR" value={`${m.input_vs_clean.psnr} dB`} hint={`L1 ${m.input_vs_clean.l1}`} />}
              {m?.output_vs_clean && <Stat label="Output PSNR" value={`${m.output_vs_clean.psnr} dB`} hint={`L1 ${m.output_vs_clean.l1}`} />}
              {m?.output_vs_input && <Stat label="Change vs input" value={`${m.output_vs_input.psnr} dB`} hint="PSNR(output, input)" />}
              <Stat label="Model" value={mode === 'universal' ? 'UDAE' : mode === 'hard' ? 'Router' : 'Soft MoE'} hint={res.model} />
            </div>
          )}

          {res && mode === 'hard' && (
            <div className="grid gap-4 lg:grid-cols-2">
              <Bars title="Classifier probabilities" values={res.probabilities} highlight={res.predicted} truth={res.true_label} />
              <div className="card flex flex-col gap-3">
                <div className="label">Routing decision</div>
                <div className="text-sm text-slate-300">Predicted corruption</div>
                <div className="text-2xl font-bold text-accent-400">{PRETTY[res.predicted]}</div>
                <div className="text-sm text-slate-300">Selected expert</div>
                <div className="text-lg font-semibold text-white">{res.selected_expert}</div>
                {res.true_label && (
                  <div className={`mt-2 rounded-lg px-3 py-2 text-sm ${res.true_label === res.predicted ? 'bg-emerald-500/10 text-emerald-300' : 'bg-amber-500/10 text-amber-300'}`}>
                    {res.true_label === res.predicted
                      ? 'Routed to the correct expert.'
                      : `Misrouted: true corruption is ${PRETTY[res.true_label]}.`}
                  </div>
                )}
              </div>
            </div>
          )}

          {res && mode === 'soft' && (
            <div className="grid gap-4 lg:grid-cols-2">
              <Bars title="Routing weights w = softmax(G(x)/τ)" values={res.weights} highlight={res.dominant_expert} truth={res.true_label} />
              <div className="card flex flex-col gap-3">
                <div className="label">Expert contributions</div>
                <div className="flex h-8 overflow-hidden rounded-lg">
                  {Object.entries(res.weights).map(([k, v], i) => (
                    <div key={k} title={`${k} ${(v * 100).toFixed(1)}%`} style={{ width: `${v * 100}%` }}
                      className={['bg-slate-400', 'bg-sky-500', 'bg-violet-500', 'bg-amber-500'][i]} />
                  ))}
                </div>
                <div className="flex flex-wrap gap-3 text-xs text-slate-300">
                  {Object.keys(res.weights).map((k, i) => (
                    <span key={k} className="flex items-center gap-1.5">
                      <span className={`h-2.5 w-2.5 rounded-sm ${['bg-slate-400', 'bg-sky-500', 'bg-violet-500', 'bg-amber-500'][i]}`} />
                      {PRETTY[k]}
                    </span>
                  ))}
                </div>
                <div className="text-sm text-slate-300">
                  Dominant branch: <span className="font-semibold text-accent-400">{PRETTY[res.dominant_expert]}</span>
                </div>
                <div className="text-sm text-slate-400">
                  Branches with ≥ 10% weight: {res.contributors.map((c) => PRETTY[c]).join(', ')}
                </div>
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
