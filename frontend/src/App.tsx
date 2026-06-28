import { useState, useEffect, useCallback } from 'react'

type Label = 'pos' | 'neg' | 'neu'
interface Result {
  text: string; compound: number; pos: number; neg: number; neu: number; label: Label
}
interface Metrics {
  total_analyzed: number; label_counts: Record<Label, number>; avg_compound: number
}

const labelMeta: Record<Label, { text: string; color: string; bar: string }> = {
  pos: { text: 'Positive', color: 'text-emerald-400', bar: 'bg-emerald-500' },
  neg: { text: 'Negative', color: 'text-rose-400', bar: 'bg-rose-500' },
  neu: { text: 'Neutral', color: 'text-slate-400', bar: 'bg-slate-500' },
}

function ScoreBar({ compound, label }: { compound: number; label: Label }) {
  const pct = Math.abs(compound) * 50
  const style: React.CSSProperties = compound >= 0
    ? { width: `${pct}%`, marginLeft: '50%' }
    : { width: `${pct}%`, marginLeft: `${50 - pct}%` }
  return (
    <div className="h-2 bg-white/10 rounded-full overflow-hidden mt-2 mb-1">
      <div
        className={`h-full rounded-full transition-all duration-500 ${labelMeta[label].bar} animate-fill-bar`}
        style={style}
      />
    </div>
  )
}

const bgByLabel: Record<string, string> = {
  pos: 'bg-green-950',
  neg: 'bg-red-950',
  neu: 'bg-blue-950',
}

function App() {
  const [input, setInput] = useState('')
  const [result, setResult] = useState<Result | null>(null)
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [history, setHistory] = useState<Result[]>(() =>
    JSON.parse(localStorage.getItem('sh') || '[]')
  )

  const fetchMetrics = useCallback(() => {
    fetch('/metrics').then(r => r.json()).then(setMetrics)
  }, [])

  useEffect(() => { fetchMetrics() }, [fetchMetrics])

  const analyze = async () => {
    const text = input.trim()
    if (!text) return
    setError(null)
    setLoading(true)
    try {
      const res = await fetch('/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })
      if (!res.ok) throw new Error(`Server returned ${res.status}`)
      const data = await res.json()
      const r = data.results[0]
      setResult(r)
      const next = [r, ...history].slice(0, 20)
      setHistory(next)
      localStorage.setItem('sh', JSON.stringify(next))
      fetchMetrics()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Analysis failed')
    } finally {
      setLoading(false)
    }
  }

  const bgClass = result ? bgByLabel[result.label] : 'bg-blue-950'

  return (
    <div className={`min-h-screen transition-colors duration-700 ${bgClass}`}>
      <div className="max-w-2xl mx-auto px-4 py-8">
      <header className="text-center mb-10">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-emerald-300 via-violet-300 to-rose-300 bg-clip-text text-transparent">
          Sentiment Analysis
        </h1>
        <p className="text-white/70 mt-2">Type text and get an instant sentiment verdict</p>
      </header>

      <div className="backdrop-blur-xl bg-white/5 rounded-2xl p-6 border border-white/10 shadow-2xl mb-6">
        <textarea
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Enter text to analyze..."
          rows={4}
          className="w-full bg-white/5 border border-white/10 rounded-xl p-4 text-white placeholder-white/40 focus:outline-none focus:ring-2 focus:ring-violet-500/50 focus:border-transparent resize-none transition"
        />
        <button
          onClick={analyze}
          disabled={!input.trim() || loading}
          className="mt-3 px-8 py-3 bg-gradient-to-r from-violet-700 to-indigo-700 hover:from-violet-600 hover:to-indigo-600 disabled:opacity-30 disabled:cursor-not-allowed rounded-xl font-semibold transition-all duration-200 active:scale-95"
        >
          {loading ? 'Analyzing…' : 'Analyze'}
        </button>
      </div>

      {error && (
        <div className="backdrop-blur-xl bg-red-950/60 rounded-2xl p-4 border border-red-500/30 shadow-2xl mb-6 animate-slide-in">
          <p className="text-red-300 text-sm">{error}</p>
        </div>
      )}

      {result && (
        <div key={result.text + result.compound} className="backdrop-blur-xl bg-white/5 rounded-2xl p-6 border border-white/10 shadow-2xl mb-6 animate-slide-in">
          <div className="flex items-center justify-between mb-2">
            <span className={`text-lg font-bold ${labelMeta[result.label].color}`}>
              {labelMeta[result.label].text}
            </span>
            <span className="text-white/70 text-sm">
              Compound: <strong className="text-white">{result.compound}</strong>
            </span>
          </div>
          <p className="text-white/80 text-sm mb-1 italic">"{result.text}"</p>
          <ScoreBar compound={result.compound} label={result.label} />
          <div className="flex gap-4 mt-3 text-xs text-white/60">
            <span>pos: {result.pos}</span>
            <span>neg: {result.neg}</span>
            <span>neu: {result.neu}</span>
          </div>
        </div>
      )}

      {metrics && (
        <div className="backdrop-blur-xl bg-white/5 rounded-2xl p-6 border border-white/10 shadow-2xl mb-6">
          <h2 className="text-lg font-semibold mb-3">Metrics</h2>
          <div className="flex gap-6 text-sm text-white/80 mb-3">
            <span>Total: <strong className="text-white">{metrics.total_analyzed}</strong></span>
            <span>Avg compound: <strong className="text-white">{metrics.avg_compound}</strong></span>
          </div>
          <div className="flex h-3 rounded-full overflow-hidden bg-white/5">
            {(['pos', 'neu', 'neg'] as const).map(l => {
              const pct = metrics.total_analyzed
                ? (metrics.label_counts[l] / metrics.total_analyzed) * 100 : 0
              return pct > 0 ? (
                <div
                  key={l}
                  className={`${labelMeta[l].bar} transition-all duration-500`}
                  style={{ width: `${pct}%` }}
                />
              ) : null
            })}
          </div>
        </div>
      )}

      {history.length > 0 && (
        <div className="backdrop-blur-xl bg-white/5 rounded-2xl p-6 border border-white/10 shadow-2xl">
          <h2 className="text-lg font-semibold mb-3">History</h2>
          <div className="space-y-2 max-h-80 overflow-y-auto [&::-webkit-scrollbar]:w-1 [&::-webkit-scrollbar-thumb]:bg-white/10 [&::-webkit-scrollbar-thumb]:rounded-full">
            {history.map((item, i) => (
              <div
                key={i}
                onClick={() => setResult(item)}
                className="flex items-center gap-3 p-3 rounded-xl bg-white/5 hover:bg-white/10 cursor-pointer transition-colors"
              >
                <span className={`w-2 h-2 rounded-full ${labelMeta[item.label].bar}`} />
                <span className={`text-xs font-semibold ${labelMeta[item.label].color} min-w-[56px]`}>
                  {labelMeta[item.label].text}
                </span>
                <span className="text-sm text-white/70 truncate">
                  {item.text.length > 60 ? item.text.slice(0, 60) + '…' : item.text}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
    </div>
  )
}

export default App
