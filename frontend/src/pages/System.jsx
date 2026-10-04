import { useEffect, useState } from 'react'
import { getJSON } from '../api.js'
import { ErrorBox, Stat } from '../components/ui.jsx'

export default function System() {
  const [health, setHealth] = useState(null)
  const [info, setInfo] = useState(null)
  const [error, setError] = useState('')

  function load() {
    setError('')
    Promise.all([getJSON('/api/health'), getJSON('/api/info')])
      .then(([h, i]) => { setHealth(h); setInfo(i) })
      .catch((e) => setError(e.message))
  }
  useEffect(load, [])

  return (
    <div className="flex flex-col gap-6">
      <header className="flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">System</h1>
          <p className="mt-1 text-sm text-slate-400">Backend health, loaded ONNX models and their input/output signatures.</p>
        </div>
        <button className="btn-ghost" onClick={load}>Refresh</button>
      </header>
      <ErrorBox error={error} />
      {health && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Stat label="Status" value={health.status} hint={`uptime ${health.uptime_s}s`} />
          <Stat label="Models loaded" value={`${health.models_loaded.length} / 7`}
            hint={Object.keys(health.models_missing).length ? `missing: ${Object.keys(health.models_missing).join(', ')}` : 'all present'} />
          <Stat label="ONNX Runtime" value={health.onnxruntime} hint={health.providers.join(', ')} />
          <Stat label="Python" value={health.python} hint={health.models_dir} />
        </div>
      )}
      {info && (
        <div className="card overflow-x-auto">
          <div className="label mb-3">ONNX models</div>
          <table className="w-full text-left text-sm">
            <thead className="text-xs uppercase text-slate-400">
              <tr><th className="py-2 pr-4">Key</th><th className="pr-4">File</th><th className="pr-4">Size</th><th className="pr-4">Inputs</th><th>Outputs</th></tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {Object.entries(info.models).map(([k, m]) => (
                <tr key={k} className="align-top">
                  <td className="py-2 pr-4 font-semibold text-accent-400">{k}</td>
                  <td className="pr-4 text-slate-300">{m.file}</td>
                  <td className="pr-4 tabular-nums text-slate-300">{m.size_mb} MB</td>
                  <td className="pr-4 font-mono text-xs text-slate-400">{m.inputs.map((i) => `${i.name} ${JSON.stringify(i.shape)}`).join(', ')}</td>
                  <td className="font-mono text-xs text-slate-400">{m.outputs.map((o) => `${o.name} ${JSON.stringify(o.shape)}`).join(', ')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {info && (
        <div className="card">
          <div className="label mb-3">Test-set corruption settings (also used by the app)</div>
          <pre className="overflow-x-auto text-xs text-slate-300">{JSON.stringify(info.corruptions.settings, null, 2)}</pre>
        </div>
      )}
    </div>
  )
}
