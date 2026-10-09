import { useEffect, useState } from 'react'
import { getJSON } from './api.js'
import Restoration from './pages/Restoration.jsx'
import Sketch from './pages/Sketch.jsx'
import System from './pages/System.jsx'

const NAV = [
  { id: 'universal', label: 'Universal Restoration', task: 'Task 1' },
  { id: 'hard', label: 'Hard-Routed Restoration', task: 'Task 2' },
  { id: 'soft', label: 'Soft Mixture-of-Experts Restoration', task: 'Task 3' },
  { id: 'sketch', label: 'Face-to-Sketch Generator', task: 'Task 4' },
  { id: 'system', label: 'System', task: 'Health & models' },
]

export default function App() {
  const [page, setPage] = useState(() => window.location.hash.slice(1) || 'universal')
  const [health, setHealth] = useState(null)

  useEffect(() => { window.location.hash = page }, [page])
  useEffect(() => {
    const tick = () => getJSON('/api/health').then(setHealth).catch(() => setHealth({ status: 'offline' }))
    tick()
    const t = setInterval(tick, 15000)
    return () => clearInterval(t)
  }, [])

  const dot = health?.status === 'ok' ? 'bg-emerald-400' : health?.status === 'degraded' ? 'bg-amber-400' : 'bg-rose-500'

  return (
    <div className="min-h-screen lg:flex">
      <nav className="border-b border-white/10 bg-ink-900/90 p-4 lg:sticky lg:top-0 lg:h-screen lg:min-h-screen lg:self-start lg:w-72 lg:shrink-0 lg:border-b-0 lg:border-r">
        <div className="mb-6 flex items-center gap-3 px-2">
          <div className="grid h-9 w-9 place-items-center rounded-xl bg-accent-500 font-bold text-ink-950">R</div>
          <div>
            <div className="font-bold text-white">RestoreLab</div>
            <div className="text-xs text-slate-400">Generative AI · Assignment 1</div>
          </div>
        </div>
        <ul className="flex gap-2 overflow-x-auto lg:flex-col">
          {NAV.map((n) => (
            <li key={n.id} className="shrink-0">
              <button onClick={() => setPage(n.id)}
                className={`w-full rounded-xl px-3 py-2.5 text-left transition ${page === n.id ? 'bg-accent-500/15 ring-1 ring-accent-500/50' : 'hover:bg-white/5'}`}>
                <div className={`text-sm font-semibold ${page === n.id ? 'text-accent-400' : 'text-slate-200'}`}>{n.label}</div>
                <div className="text-xs text-slate-500">{n.task}</div>
              </button>
            </li>
          ))}
        </ul>
        <div className="mt-6 hidden items-center gap-2 rounded-xl bg-ink-800/60 px-3 py-2 text-xs text-slate-300 lg:flex">
          <span className={`h-2 w-2 rounded-full ${dot}`} />
          API {health?.status || '…'} · {health?.models_loaded?.length ?? 0}/7 models
        </div>
      </nav>
      <main className="flex-1 p-4 sm:p-8">
        {page === 'sketch' ? <Sketch /> : page === 'system' ? <System /> : <Restoration key={page} mode={page} />}
      </main>
    </div>
  )
}
